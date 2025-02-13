from pytorch_lightning import Trainer
from monoscene.models.monoscene import MonoScene
from monoscene.data.kitti_360.kitti_360_dm import Kitti360DataModule
import hydra
from omegaconf import DictConfig
import torch
import shutil
import os
from hydra.utils import get_original_cwd


# @hydra.main(config_name="../config/monoscene.yaml")
@hydra.main(config_name="../config/monoscene-kitti360.yaml")
def main(config: DictConfig):

    # 复制下载好的ckpt到指定缓存位置
    # 源文件路径
    source_path = '/workspace/mnt/storage/shihao/MyCode-02/SSCBench/method/MonoScene/trained_models/tf_efficientnet_b7_ns-1dbc32de.pth'
    # 目标文件路径
    destination_path = '/root/.cache/torch/hub/checkpoints/tf_efficientnet_b7_ns-1dbc32de.pth'
    # 确保目标目录存在
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    # 复制文件
    shutil.copy(source_path, destination_path)

    torch.set_grad_enabled(False)
    config.batch_size = 1
    n_classes = 19
    feature = 64
    # n_gpus=1 
    # batch_size=1
    project_scale = 2
    full_scene_size = (256, 256, 32)
    data_module = Kitti360DataModule(
        root=config.kitti_360_root,
        preprocess_root=config.kitti_360_preprocess_root,
        frustum_size=config.frustum_size,
        batch_size=int(config.batch_size / config.n_gpus),
        num_workers=8,      # 0
        #num_workers=int(config.num_workers_per_gpu * config.n_gpus),
    )

    # elif config.dataset == "NYU":
    #     config.batch_size = 2
    #     project_scale = 1
    #     n_classes = 12
    #     feature = 200
    #     full_scene_size = (60, 36, 60)
    #     data_module = NYUDataModule(
    #         root=config.NYU_root,
    #         preprocess_root=config.NYU_preprocess_root,
    #         n_relations=config.n_relations,
    #         frustum_size=config.frustum_size,
    #         batch_size=int(config.batch_size / config.n_gpus),
    #         num_workers=int(config.num_workers_per_gpu * config.n_gpus),
    #     )

    trainer = Trainer(
        sync_batchnorm=True, deterministic=True, gpus=config.n_gpus, accelerator="ddp"
    )

    # model_path = '/home/duan/shl/Benchmark/MonoScene/logdir/kitti360/exp_kitti_360_1_FrusSize_8_nRelations4_WD0.0001_lr0.0001_CEssc_geoScalLoss_semScalLoss_fpLoss_CERel_3DCRP_Proj_2_4_8/checkpoints/last.ckpt'
    model_path = '/workspace/mnt/storage/shihao/MyCode-02/SSCBench/method/MonoScene/trained_models/Monoscene_kitti360.ckpt'
    model = MonoScene.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
    )
    model.eval()
    data_module.setup()
    # val_dataloader = data_module.val_dataloader()
    val_dataloader = data_module.test_dataloader()
    trainer.test(model, test_dataloaders=val_dataloader)


if __name__ == "__main__":
    main()
