"""
Probabilistic Detectron Training Script following Detectron2 training script found at detectron2/tools.
"""
import core
import os
import sys

# This is very ugly. Essential for now but should be fixed.
# 所以，这行代码的作用是将一个特定的路径添加到 Python 解释器的路径列表中，
# 这个路径是通过连接核心模块的顶级目录路径、'src' 目录和 'detr' 目录得到的。
# 这样做的目的可能是为了让 Python 解释器能够找到位于 'src/detr' 目录下的模块或文件。
sys.path.append(os.path.join(core.top_dir(), 'src', 'detr'))

# Detectron imports
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results


# Project imports
from core.setup import setup_config, setup_arg_parser
from default_trainer import DefaultTrainer



class Trainer(DefaultTrainer):
    # 这段代码是一个自定义的 Detectron2 训练器类（Trainer），它继承自 Detectron2 的 DefaultTrainer 类。
    # 在这个自定义的 Trainer 类中，有一个名为 build_evaluator 的类方法，用于构建用于训练后的 mAP（平均精度平均值）报告的评估器。
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Builds evaluators for post-training mAP report. 这是一个类方法，用于构建评估器。它接受三个参数：
        Args:
            cls 表示类本身
            cfg(CfgNode): a detectron2 CfgNode ， cfg 是一个 Detectron2 的配置节点（CfgNode）
            dataset_name(str): registered dataset name ， dataset_name 是已注册的数据集名称。

        Returns:
            detectron2 DatasetEvaluators object
        """
        # 这一行代码用于创建输出文件夹的路径，该文件夹用于存储推理结果。
        # cfg.OUTPUT_DIR 是训练过程中模型和日志输出的主文件夹，因此在这里创建一个子文件夹 "inference"。
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        #这一行代码创建了一个评估器列表。在这个例子中，只有一个评估器，即 COCOEvaluator。
        # COCOEvaluator 是用于 COCO 数据集的评估器，用于评估模型在测试集上的性能。
        # 它接受数据集名称、配置节点、是否使用小型评估模式（默认为 False）和输出文件夹路径作为参数。
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        #这个方法返回一个 DatasetEvaluators 对象，其中包含了构建好的评估器列表。
        # DatasetEvaluators 是 Detectron2 中用于批量执行评估器的对象，它接受一个评估器列表作为参数。
        return DatasetEvaluators(evaluators)



def main(args):
    # Setup config node
    cfg = setup_config(args,
                       random_seed=args.random_seed, is_testing=False, ood=False)
    # For debugging only
    #cfg.defrost()
    #cfg.DATALOADER.NUM_WORKERS = 0
    #cfg.SOLVER.IMS_PER_BATCH = 1

    # Eval only mode to produce mAP results
    # Build Trainer from config node. Begin Training.

    trainer = Trainer(cfg)

    if args.eval_only:
        # 检查脚本是否处于仅评估模式。如果是，则加载预训练模型，执行评估，并返回结果而不进行训练。
        model = trainer.build_model(cfg)
        model.eval()
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    # Create arg parser 设置参数解析器。
    arg_parser = setup_arg_parser()

    args = arg_parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
