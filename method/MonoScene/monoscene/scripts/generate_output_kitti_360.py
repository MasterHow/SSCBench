from pytorch_lightning import Trainer
from monoscene.models.monoscene import MonoScene
from monoscene.data.NYU.nyu_dm import NYUDataModule
from monoscene.data.semantic_kitti.kitti_dm import KittiDataModule
from monoscene.data.kitti_360.kitti_360_dm_vis import Kitti360DataModule
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os
from hydra.utils import get_original_cwd
from tqdm import tqdm
import pickle
import yaml


def get_inv_remap_lut_from_yaml(yaml_path):
    """

    Args:
        yaml_path: path to dataset config.

    Returns:
        a map function that mapping learning map to class map.
    by Hao.
    """
    # 读取YAML文件
    with open(yaml_path, 'r') as file:
        dataset_config = yaml.safe_load(file)

    # 获取learning_map_inv部分
    learning_map_inv = dataset_config['learning_map_inv']

    # 获取最大的键值
    maxkey = max(learning_map_inv.keys())

    # 创建一个足够大的lookup table，以防出现未知标签
    remap_lut = np.zeros((maxkey + 1), dtype=np.int32)

    # 使用learning_map_inv中的键和值填充lookup table
    remap_lut[list(learning_map_inv.keys())] = list(learning_map_inv.values())

    return remap_lut


@hydra.main(config_name="../config/monoscene-kitti360.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)

    # Setup dataloader
    if config.dataset == "kitti" or config.dataset == "kitti_360":
        feature = 64
        project_scale = 2
        full_scene_size = (256, 256, 32)

        if config.dataset == "kitti":
            data_module = KittiDataModule(
                root=config.kitti_root,
                preprocess_root=config.kitti_preprocess_root,
                frustum_size=config.frustum_size,
                batch_size=int(config.batch_size / config.n_gpus),
                num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            )
            data_module.setup()
            data_loader = data_module.val_dataloader()
            # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
        else:
            data_module = Kitti360DataModule(
                root=config.kitti_360_root,
                sequences=[config.kitti_360_sequence],
                # n_scans=2000,
                n_scans=2000000,
                # batch_size=1,
                batch_size=int(config.batch_size / config.n_gpus),
                # num_workers=0,
                num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            )
            data_module.setup()
            data_loader = data_module.dataloader()

            # 获取反映射函数 用于存储为.label标签
            yaml_path = "/workspace/mnt/storage/shihao/MyCode-02/SSCBench/dataset/configs/kitti360.yaml"
            inv_remap_lut = get_inv_remap_lut_from_yaml(yaml_path)

    elif config.dataset == "NYU":
        project_scale = 1
        feature = 200
        full_scene_size = (60, 36, 60)
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )
        data_module.setup()
        data_loader = data_module.val_dataloader()
        # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
    else:
        print("dataset not support")

    # Load pretrained models
    if config.dataset == "NYU":
        model_path = os.path.join(
            get_original_cwd(), "trained_models", "monoscene_nyu.ckpt"
        )
    else:
        # model_path = os.path.join(
        #     get_original_cwd(), "trained_models", "monoscene_kitti.ckpt"
        # )
        model_path = config.ckpt_path
    print("Loading model from {}".format(model_path))

    model = MonoScene.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
    )
    model.cuda()
    model.eval()

    # Save prediction and additional data 
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(config.output_path, config.dataset)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch["img"] = batch["img"].cuda()
            pred = model(batch)
            y_pred = torch.softmax(pred["ssc_logit"], dim=1).detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)

            # for i in range(config.batch_size):      # 这里应该改成获取y_pred的Batch size维度
            # 使用y_pred的batch size
            for i in range(y_pred.shape[0]):
                out_dict = {"y_pred": y_pred[i].astype(np.uint16)}
                if "target" in batch:
                    out_dict["target"] = (
                        batch["target"][i].detach().cpu().numpy().astype(np.uint16)
                    )

                if config.dataset == "NYU":
                    write_path = output_path
                    filepath = os.path.join(write_path, batch["name"][i] + ".pkl")
                    out_dict["cam_pose"] = batch["cam_pose"][i].detach().cpu().numpy()
                    out_dict["vox_origin"] = (
                        batch["vox_origin"][i].detach().cpu().numpy()
                    )
                else:
                    write_path = os.path.join(output_path, batch["sequence"][i])
                    # filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                    # filepath = os.path.join(write_path, batch["frame_id"][i] + ".npy")
                    filepath = os.path.join(write_path, batch["frame_id"][i] + ".label")
                    out_dict["fov_mask_1"] = (
                        batch["fov_mask_1"][i].detach().cpu().numpy()
                    )
                    out_dict["cam_k"] = batch["cam_k"][i].detach().cpu().numpy()
                    out_dict["T_velo_2_cam"] = (
                        batch["T_velo_2_cam"][i].detach().cpu().numpy()
                    )

                os.makedirs(write_path, exist_ok=True)
                with open(filepath, "wb") as handle:
                    # pickle.dump(out_dict, handle)
                    # np.save(filepath, out_dict["y_pred"])   # 256*256*32 的npy文件

                    pred_to_save = out_dict["y_pred"]
                    # 改变形状方便存储为一维文件
                    pred_to_save = np.moveaxis(pred_to_save, [0, 1, 2], [0, 2, 1]).reshape(
                        -1).astype(np.uint16)       # 256*256*32 => 256*32*256 => -1
                    # 从学习标签反映射回class标签
                    pred_to_save = inv_remap_lut[pred_to_save].astype(np.uint16)
                    # 保存.label文件
                    pred_to_save.tofile(filepath)

                    print("wrote to", filepath)


if __name__ == "__main__":
    main()
