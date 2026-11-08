import warnings
from datetime import datetime
from functions.set_seed import *
from new_model.Multi_Layer_forum_8 import  main as run_forum
from functions.dataload import search_specify_data_from_dataname_for_halo,search_specify_data_from_dataname_for_forum,read_exist_substructure_to_datatxt_for_forum,read_exist_substructure_to_datatxt_for_halo
import argparse

# 忽略特定的UserWarning
warnings.filterwarnings("ignore")


"""该启动程序主要应用于启动程序"""
def stated_train_run(args):
    # 从命令行参数获取配置
    dataname = args.dataname
    specify_data = args.specify_data
    specify_number = args.specify_number
    storage_number = args.storage_number
    dataset_root = args.dataset_root
    project_root = args.project_root

    start_time = datetime.now()
    print("First Model training started at:", start_time)
    if project_root=="":
        project_root = os.path.dirname(os.path.abspath(__file__))

    if dataset_root=="":
        # 数据集的路径
        dataset_root = project_root + f"/dataset/{dataname}"
        # dataset_root = f"/root/autodl-tmp/project/data/{dataname}"

    # 构建dataset目录的绝对路径
    data_root = os.path.join(project_root, "result")

    if specify_data:
        '''下面是使用指定的811数据集'''
        if dataname == "halo":
            train_tree_set, val_tree_set, test_data_list, test_one_list = search_specify_data_from_dataname_for_halo(
                dataname, specify_number)
        else:
            train_tree_set, val_tree_set, test_data_list, test_one_list = search_specify_data_from_dataname_for_forum(
                dataname, specify_number)
        # root = f"/root/autodl-tmp/project/result_v8/{dataname}/specified_dataset{specify_number}/"
        root = data_root + f"/{dataname}/specified_dataset{specify_number}/"

    else:
        '''使用结构划分数据集'''
        if dataname == "halo":
            train_tree_set, val_tree_set, test_data_list, test_one_list = read_exist_substructure_to_datatxt_for_halo(
                dataname,dataset_root, train_ratio=0.8, val_ratio=0.1, record=1, specify_number=specify_number)
        else:
            train_tree_set, val_tree_set, test_data_list, test_one_list = read_exist_substructure_to_datatxt_for_forum(
                dataname,dataset_root, train_ratio=0.8, val_ratio=0.1, record=1, specify_number=specify_number)
        # root = f"/root/autodl-tmp/project/result_v8/{dataname}/split_dataset/"
        root = data_root + f"/{dataname}/split_dataset/"

    storage_path = root + f"no.{storage_number}_result.txt"
    storage_loss_path = root + f"no.{storage_number}_loss.txt"
    storage_test_path = root + f"no.{storage_number}_test_one.txt"

    run_forum(dataname,train_tree_set, val_tree_set, test_data_list, test_one_list, storage_path, storage_loss_path, storage_test_path)

    end_time = datetime.now()
    print("First Model training ended at:", end_time)
    # 计算耗时
    elapsed_time = end_time - start_time
    print("First Model training elapsed time:", elapsed_time)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Model training launcher")

    # Core parameters
    parser.add_argument("--dataname", type=str, default="forum",
                        help="Dataset name (forum/novel/halo)")
    parser.add_argument("--specify_data", type=bool, default=False,
                        help="Whether to use specified dataset (default: False)")
    parser.add_argument("--specify_number", type=str, default="2",
                        help="Specified dataset number")
    parser.add_argument("--storage_number", type=str, default="Test001",
                        help="Result file number")

    # Path parameters
    parser.add_argument("--project_root", type=str, default="",
                        help="Project root absolute path (auto-calculated by default)")
    parser.add_argument("--dataset_root", type=str, default="",
                        help="Dataset directory absolute path (uses dataset/ in project by default)")


    # Training parameters
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--record", type=int, default=1,
                        help="Whether to record dataset split (1=yes, 0=no)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)
    stated_train_run(args)







