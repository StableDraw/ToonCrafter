import io
import os
import sys
import time
from omegaconf import OmegaConf
from PIL import Image
import torch
import numpy
from torchvision.utils import make_grid, _log_api_usage_once
from scripts.evaluation.funcs import load_model_checkpoint, batch_ddim_sampling
from utils.utils import instantiate_from_config
from huggingface_hub import hf_hub_download
from einops import repeat
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from einops import rearrange

import av

sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))


def tensor_to_binary_image_list(video_array):
    """
    Функция, преобразующая 4D тензор изображений в список бинарных изображений
    """

    binary_image_list = []
    for i in range(video_array.size(0)): 
        image = Image.fromarray(video_array[i, :, :, :].numpy())
        buf = io.BytesIO()
        image.save(buf, format = "PNG")
        b_data = buf.getvalue()
        image.close
        binary_image_list.append(b_data)

    return binary_image_list


def image_array_to_binary_video(video, fps, video_codec = "libx264", is_image_list = False, options = None, audio_array = None, audio_fps = None, audio_codec = None, audio_options = None):
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        video (Tensor[T, H, W, C]): tensor containing the individual frames or binary image list,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        is_image_list (bool): is input image list or 4d image tensor
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
    """

    if is_image_list == False:
        video_array = torch.as_tensor(video, dtype = torch.uint8).numpy()
    else:
        video_array = []
        for frame in video:
            video_array.append(numpy.asarray(Image.open(io.BytesIO(frame)).convert("RGB")))
        video_array = numpy.array(video_array)

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = numpy.round(fps)

    binary_video = io.BytesIO()

    with av.open(binary_video, mode = "w", format = "mp4") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate = audio_fps)
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = container.streams.audio[0].format.name

            format_dtype = numpy.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)

            frame = av.AudioFrame.from_ndarray(audio_array, format = audio_sample_fmt, layout=audio_layout)

            frame.sample_rate = audio_fps

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format = "rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)

    return binary_video.getbuffer()


def postprocess_video(batch_tensors, return_video = False, fps = 10):
    # b,samples,c,t,h,w
    n_samples = batch_tensors.shape[1]
    for vid_tensor in batch_tensors:
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(2, 0, 1, 3, 4) # t,n,c,h,w
        frame_grids = [make_grid(framesheet, nrow = int(n_samples)) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, n*h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        binary = tensor_to_binary_image_list(grid)

        if return_video == True:
            binary = image_array_to_binary_video(binary, fps = fps, video_codec = 'h264', is_image_list = True, options = {'crf': '10'})  

    return binary


def weights_download(opt):

    REPO_ID = 'Doubiiu/ToonCrafter'
    filename_list = ['model.ckpt']
    if not os.path.exists(opt["ckpt_path"][:opt["ckpt_path"].rfind("/")]):
        os.makedirs(opt["ckpt_path"][:opt["ckpt_path"].rfind("/")])
    for filename in filename_list:
        local_file = opt["ckpt_path"]
        if not os.path.exists(local_file):
            hf_hub_download(repo_id = REPO_ID, filename = filename, local_dir = opt["ckpt_path"][:opt["ckpt_path"].rfind("/")], local_dir_use_symlinks = False)


def get_latent_z_with_hidden_states(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    encoder_posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)

    hidden_states_first_last = []
    ### use only the first and last hidden states
    for hid in hidden_states:
        hid = rearrange(hid, '(b t) c h w -> b c t h w', t = t)
        hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim = 2)
        hidden_states_first_last.append(hid_new)

    z = model.get_first_stage_encoding(encoder_posterior).detach()
    z = rearrange(z, '(b t) c h w -> b c t h w', b = b, t = t)
    return z, hidden_states_first_last


def animate_images(first_binary_data, second_binary_data, prompt, opt, return_video = False, gpu_num = 1):

    image = numpy.asarray(Image.open(io.BytesIO(first_binary_data)).convert("RGB"))
    image2 = numpy.asarray(Image.open(io.BytesIO(second_binary_data)).convert("RGB"))

    seed_everything(opt["seed"])
    resolution = (args["height"], args["width"]) #hw
    transform = transforms.Compose([
        transforms.Resize(min(resolution)),
        transforms.CenterCrop(resolution),
        ])

    torch.cuda.empty_cache()
    print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    start = time.time()
    gpu_id = 0

    config = OmegaConf.load(opt["config"])
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint']=False   

    model_list = []
    for gpu_id in range(gpu_num):
        model = instantiate_from_config(model_config)
        # model = model.cuda(gpu_id)
        print(opt["ckpt_path"])
        assert os.path.exists(opt["ckpt_path"]), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, opt["ckpt_path"])
        model.eval()
        model_list.append(model)
    model_list = model_list

    model = model_list[gpu_id]
    model = model.cuda()
    batch_size = opt["bs"]
    channels = model.model.diffusion_model.out_channels
    #frames = model.temporal_length
    frames = opt["video_length"]
    h, w = resolution[0] // 8, resolution[1] // 8
    noise_shape = [batch_size, channels, frames, h, w]

    # text cond
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_emb = model.get_learned_conditioning([prompt])

        # img cond
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
        img_tensor = (img_tensor / 255. - 0.5) * 2

        image_tensor_resized = transform(img_tensor) #3,h,w
        videos = image_tensor_resized.unsqueeze(0).unsqueeze(2) # bc1hw
            
        # z = get_latent_z(model, videos) #bc,1,hw
        videos = repeat(videos, 'b c t h w -> b c (repeat t) h w', repeat = frames // 2)

        img_tensor2 = torch.from_numpy(image2).permute(2, 0, 1).float().to(model.device)
        img_tensor2 = (img_tensor2 / 255. - 0.5) * 2
        image_tensor_resized2 = transform(img_tensor2) #3,h,w
        videos2 = image_tensor_resized2.unsqueeze(0).unsqueeze(2) # bchw
        videos2 = repeat(videos2, 'b c t h w -> b c (repeat t) h w', repeat = frames // 2)
            
        videos = torch.cat([videos, videos2], dim = 2)
        z, hs = get_latent_z_with_hidden_states(model, videos)

        img_tensor_repeat = torch.zeros_like(z)

        img_tensor_repeat[:,:,:1,:,:] = z[:,:,:1,:,:]
        img_tensor_repeat[:,:,-1:,:,:] = z[:,:,-1:,:,:]

        cond_images = model.embedder(img_tensor.unsqueeze(0)) ## blc
        img_emb = model.image_proj_model(cond_images)

        imtext_cond = torch.cat([text_emb, img_emb], dim = 1)

        fs = torch.tensor([opt["frame_stride"]], dtype = torch.long, device = model.device)
        cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
            
        ## inference
        batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples = opt["n_samples"], ddim_steps = opt["ddim_steps"], ddim_eta = opt["ddim_eta"], cfg_scale = opt["unconditional_guidance_scale"], hs = hs)

        ## remove the last frame
        if image2 is None:
            batch_samples = batch_samples[:,:,:,:-1,...]
        ## b,samples,c,t,h,w
        prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
        prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
        prompt_str=prompt_str[:40]
        if len(prompt_str) == 0:
            prompt_str = 'empty_prompt'

    binary_video = postprocess_video(batch_samples, return_video = return_video, fps = opt["save_fps"])
    torch.cuda.empty_cache()

    return binary_video


if __name__ == "__main__":

    args = {
        "ckpt_path": "weights/model.ckpt", #checkpoint path
        "config": "configs/inference_512_v1.0.yaml", #config (yaml) path
        "n_samples": 1, #num of samples per prompt
        "ddim_steps": 50, #steps of ddim if positive, otherwise use DDPM
        "ddim_eta": 1.0, #eta for ddim sampling (0.0 yields deterministic sampling)
        "bs": 1, #batch size for inference, should be one
        "height": 320, #image height, in pixel space
        "width": 512, #image width, in pixel space
        "frame_stride": 10, #frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)
        "unconditional_guidance_scale": 7.5, #prompt classifier-free guidance
        "seed": 123, #seed for seed_everything
        "video_length": 16, #inference video length
        "save_fps": 8, #fps for saving
        #not implemeted yet [todo]
        "negative_prompt": False, #negative prompt
        "multiple_cond_cfg": False, #use multi-condition cfg or not
        "cfg_img": None, #guidance scale for image conditioning
        "timestep_spacing": "uniform_trailing", #The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        "guidance_rescale": 0.7, #guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)
        "perframe_ae": True, #if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024

        ## currently not support looping video and generative frame interpolation
        "loop": False, #generate looping videos or not
        "interp": False #generate generative frame interpolation or not
    }

    with open("frame1.png", "rb") as f:
        first_binary_data = f.read()

    with open("frame2.png", "rb") as f:
        second_binary_data = f.read()

    prompt = "an anime sketch"

    return_video = True #Возвращать видео или список изображений кадров. Если False - будет возвращён список изображений

    #weights_download(args) #Необязательная функция. Нужна для загрузки весов после установки
    torch.cuda.empty_cache()

    binary = animate_images(first_binary_data, second_binary_data, prompt, args, return_video = return_video)

    if return_video == False:
        for i, img in enumerate(binary):
            Image.open(io.BytesIO(img)).save(f"result/{i}.png")
    else:
        # Write BytesIO from RAM to file, for testing
        with open("output.mp4", "wb") as f:
            f.write(binary)