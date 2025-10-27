import PIL.Image
import trimesh
import options
import torch
import numpy as np
from model import Classify_Network,Shape_Network,Texture_Network
import os
import clip
import dnnlib
import cv2
import torchvision
from torchvision.transforms.transforms import ToPILImage
import legacy
from torch import optim
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)
from innovations import MultiViewRenderer, ProgressiveOptimizer, QualityEvaluator


#to replace trimesh.load
def load_ori_mesh(fn):
    return trimesh.load(fn,resolver=None,split_object=False,group_material=False,skip_materials=False,maintain_order=True,process=False)


class CLIPLoss(torch.nn.Module):
    def __init__(self, stylegan_size=512):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image, text):
        image_upsam = self.upsample(image)
        image = self.avg_pool(image_upsam)
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity

class generation(torch.nn.Module):
    def __init__(self):
        super(generation, self).__init__()

def encode_text(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    tokens = clip.tokenize(text,truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens).float()
    return text_features

def gen_onehot(opt,text):

    classify_model = Classify_Network.MLP()
    state = torch.load(opt.classfier_path)
    classify_model.load_state_dict(state['MLP'])
    classify_model.cuda()
    classify_model.eval()

    text_features = encode_text(text)

    onehot_pred = classify_model(text_features).reshape(24,8)
    shape_onehot = onehot_pred[:16,:]
    shape_onehot = ((shape_onehot==shape_onehot.max(dim=-1,keepdim=True)[0])*1).reshape(1,-1)
    texture_onehot = torch.cat((onehot_pred[:3,:],onehot_pred[16:,:]),dim=0)
    texture_onehot = ((texture_onehot==texture_onehot.max(dim=-1,keepdim=True)[0])*1).reshape(1,-1)
    all_pred = torch.cat((shape_onehot.reshape(-1,8),texture_onehot.reshape(-1,8)[3:,:]),dim=0)

    return all_pred,shape_onehot,texture_onehot

def gen_shape(shape_label,opt):

    device = torch.device("cuda:0")
    model = Shape_Network.MLP()
    state = torch.load(opt.ShapeNet_path)
    model.load_state_dict(state['MLP'])
    model.to(device)
    model.eval()
    mean_mesh = load_ori_mesh("./predef/mean_face_3DMM_300.obj")
    # mean_verts = np.load("./predef/mean_verts.npy")
    core = np.load("./predef/core_1627_300_weight_10.npy")
    pred_param = model(shape_label).detach().cpu().numpy()
    pred_verts = np.matmul(pred_param,core).reshape(-1,3)
    curr_mesh = mean_mesh.copy()
    curr_mesh.vertices = mean_mesh.vertices + pred_verts

    # curr_mesh.export("./result/0_1.obj");

    return curr_mesh,torch.from_numpy(pred_param).cuda()

def gen_texture(opt,texture_label):
    trans_pil = ToPILImage()
    device = torch.device('cuda')
    with dnnlib.util.open_url(opt.TextureNet_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    Mapping = G.mapping
    Synthesis = G.synthesis
    z = torch.randn(1, G.z_dim).to(device)
    ws = Mapping(z, texture_label)
    img = Synthesis(ws, noise_mode='const')
    texture = torch.clip((img[0]+1)/2,0,1)
    # texture = texture.squeeze(0)  # 压缩一维
    texture = trans_pil(texture)
    # texture.save("./result/material_0.png")

    return texture,ws,Synthesis

def concrete_synthesis(opt,shape_label,texture_label):

    mesh,pred_param = gen_shape(shape_label,opt)
    texture,ws,Synthesis = gen_texture(opt,texture_label=texture_label)

    mesh.visual.material.image = texture

    save_path = os.path.join(opt.result_dir,opt.name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    obj_save_path = os.path.join(save_path,"result_concrete.obj")
    mesh.export(obj_save_path);

    return ws,pred_param,Synthesis


def diff_render(render_img, curr_verts):
    device = "cuda"
    mean_verts = torch.from_numpy(np.load("./predef/mean_verts.npy")).cuda()
    faces = torch.from_numpy(np.load("./predef/faces.npy")).to(torch.int64).cuda()
    
    vertices = (curr_verts + mean_verts).unsqueeze(0)
    faces_tensor = faces.unsqueeze(0)
    
    num_verts = vertices.shape[1]
    texture_rgb = (render_img[0] + 1) / 2
    texture_rgb = texture_rgb.permute(1, 2, 0).reshape(-1, 3)
    
    if texture_rgb.shape[0] < num_verts:
        texture_rgb = torch.nn.functional.interpolate(
            texture_rgb.unsqueeze(0).permute(0, 2, 1),
            size=num_verts,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1).squeeze(0)
    elif texture_rgb.shape[0] > num_verts:
        indices = torch.linspace(0, texture_rgb.shape[0] - 1, num_verts).long()
        texture_rgb = texture_rgb[indices]
    
    verts_rgb = texture_rgb.unsqueeze(0)
    textures = TexturesVertex(verts_features=verts_rgb)
    mesh = Meshes(verts=vertices, faces=faces_tensor, textures=textures)
    
    R, T = look_at_view_transform(dist=2.7, elev=0, azim=0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )
    
    images = renderer(mesh)
    img_pred = images[0, ..., :3]
    
    return img_pred


def prompt_synthesis(ws,params,Synthesis):
    trans_pil = ToPILImage()
    latent_code_int = ws
    param_init = params

    latent = latent_code_int.detach().clone()
    latent.requires_grad = True
    param = param_init.detach().clone()
    param.requires_grad = True
    core = torch.from_numpy(np.load("./predef/core_1627_300_weight_10.npy")).cuda()

    clip_loss = CLIPLoss()

    # 创新点2：初始化渐进式优化器
    progressive_opt = ProgressiveOptimizer(
        total_steps=opt.step,
        initial_lr_latent=opt.lr_latent,
        initial_lr_param=opt.lr_param,
        initial_lambda_latent=opt.lambda_latent,
        initial_lambda_param=opt.lambda_param
    )
    
    latent_optimizer = optim.Adam([latent], lr=opt.lr_latent)
    params_optimizer = optim.Adam([param], lr=opt.lr_param)

    # 创新点1：初始化多视角渲染器
    multi_view_renderer = MultiViewRenderer(device="cuda", image_size=512)
    
    # 创新点3：初始化质量评估器
    curr_save_folder = os.path.join(opt.result_dir, opt.name, opt.prompt if opt.prompt else "default")
    if not os.path.exists(curr_save_folder):
        os.makedirs(curr_save_folder, exist_ok=True)
    quality_evaluator = QualityEvaluator(save_dir=curr_save_folder)

    pbar = tqdm(range(opt.step))
    
    current_stage = ""

    for i in pbar:
        
        # 创新点2：获取当前阶段的超参数
        stage_params = progressive_opt.get_current_params(i)
        
        # 如果进入新阶段，更新优化器学习率
        if stage_params['stage'] != current_stage:
            current_stage = stage_params['stage']
            for param_group in latent_optimizer.param_groups:
                param_group['lr'] = stage_params['lr_latent']
            for param_group in params_optimizer.param_groups:
                param_group['lr'] = stage_params['lr_param']

        img_gen = Synthesis(latent)  ## 1*3*512*512

        curr_verts = torch.matmul(param,core).reshape(-1,3)

        render_img = img_gen.permute(0,2,3,1)

        # 创新点1：使用多视角渲染（主要使用前视图进行优化）
        img_pred = multi_view_renderer.render_multi_view(curr_verts, render_img, 'front')
        img_rgb = img_pred.detach().cpu().numpy()[:,:,[2,1,0]]*255
        img_chw = img_pred.unsqueeze(0).permute(0,3,1,2)

        text_tokens = clip.tokenize(opt.prompt).cuda()
        c_loss = clip_loss(img_chw,text_tokens)
        l2_loss_latent = ((latent-latent_code_int)**2).sum()
        l2_loss_param = ((param-params)**2).sum()

        # 创新点1：添加多视角一致性损失（可选，通过配置开关）
        if opt.use_multi_view and i % 5 == 0:  # 每5步计算一次以节省时间
            consistency_loss = multi_view_renderer.compute_multi_view_consistency_loss(curr_verts, render_img)
            consistency_weight = 0.1
        else:
            consistency_loss = 0.0
            consistency_weight = 0.0

        # 使用渐进式权重
        loss = (c_loss + 
                stage_params['lambda_latent'] * l2_loss_latent + 
                stage_params['lambda_param'] * l2_loss_param +
                consistency_weight * consistency_loss)

        latent_optimizer.zero_grad()
        params_optimizer.zero_grad()
        loss.backward()
        latent_optimizer.step()
        params_optimizer.step()

        # 创新点3：评估质量并保存最佳结果
        is_best, quality_score = quality_evaluator.evaluate(
            i, c_loss, l2_loss_latent, l2_loss_param, loss
        )
        
        if is_best:
            quality_evaluator.save_best_state(latent, param, i)

        pbar.set_description(
            (
                f"{current_stage} | loss: {loss.item():.4f} | quality: {quality_score:.4f}"
            )
        )

        if opt.save_step > 0 and i % opt.save_step == 0:
            with torch.no_grad():
                img_gen = Synthesis(latent)

            torchvision.utils.save_image(img_gen,f"{opt.inter_dir}/texture_map/{str(i).zfill(5)}_tex.jpg", normalize=True)
            cv2.imwrite(os.path.join(opt.inter_dir,"render",f"{str(i).zfill(5)}_render.jpg"),img_rgb)

    # 创新点3：生成优化报告
    report = quality_evaluator.generate_report()
    print(f"\n=== 优化报告 ===")
    print(f"最佳迭代: {report['best_iteration']}")
    print(f"最佳质量分数: {report['best_score']:.4f}")
    print(f"报告已保存至: {curr_save_folder}")
    
    # 加载最佳结果
    best_state = torch.load(os.path.join(curr_save_folder, 'best_model.pth'))
    latent = best_state['latent'].cuda()
    param = best_state['param'].cuda()
    print(f"已加载最佳结果（迭代 {best_state['iteration']}）")

    img_final = Synthesis(latent)
    # texture_final = PIL.Image.fromarray(np.clip(np.uint8(((img_final[0].permute(1,2,0).detach().cpu().numpy()+1)/2)*255),0,255))
    texture_final = trans_pil(torch.clip((img_final[0]+1)/2,0,1))


    verts_final = torch.matmul(param,core).reshape(-1,3).detach().cpu().numpy()
    mean_mesh = load_ori_mesh("./predef/mean_face_3DMM_300.obj")
    final_mesh = mean_mesh.copy()
    final_mesh.visual.material.image = texture_final
    final_mesh.vertices = mean_mesh.vertices + verts_final

    final_mesh.export(os.path.join(curr_save_folder,"result_prompt.obj"));
    
    # 创新点1：保存多视角渲染结果
    if opt.save_multi_view:
        print("\n生成多视角渲染图像...")
        with torch.no_grad():
            curr_verts = torch.matmul(param,core).reshape(-1,3)
            render_img = img_final.permute(0,2,3,1)
            
            for view_name in ['front', 'left', 'right', 'top_left', 'top_right']:
                view_img = multi_view_renderer.render_multi_view(curr_verts, render_img, view_name)
                view_rgb = view_img.detach().cpu().numpy()[:,:,[2,1,0]]*255
                cv2.imwrite(os.path.join(curr_save_folder, f"view_{view_name}.jpg"), view_rgb)
        print(f"多视角图像已保存至: {curr_save_folder}")


def gen_full_mesh(opt):

    ## Text Parser: generate ont-hot code
    all_label,shape_label,texture_label = gen_onehot(opt,text=opt.descriptions)

    ## Concrete Synthesis
    ws, pred_param, Synthesis = concrete_synthesis(opt,shape_label,texture_label)

    ## Abstract Synthesis
    if opt.prompt:
        prompt_synthesis(ws,pred_param,Synthesis)


if __name__ == '__main__':
    opt = options.Options().parse()
    gen_full_mesh(opt)



