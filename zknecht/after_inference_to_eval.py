import os
from loguru import logger


distributed_rank = 0
filename = "training_log.txt"
format = f"[Rank #{distributed_rank}] | " + "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
VAL_TXT = [
    "/home/sun/Bev/BeMapNet/assets/splits/argoverse2/geo_near_splits/av2_geosplit_val_interval_1.txt",
    # "/home/sun/Bev/BeMapNet/assets/splits/argoverse2/av2_origin_val_list.txt"
]

gt_dir = "/home/sun/Bev/BeMapNet/data/argoverse2/geosplits_interval_1"
output_dir = '/home/sun/Bev/BeMapNet/outputs/bemapnet_av2_res50/2024-09-12'
dt_dir = os.path.join(output_dir, "evaluation", "results")
save_file = os.path.join(output_dir, "eval.log")

logger.add(
    save_file,
    format=format,
    filter="",
    level="INFO" if distributed_rank == 0 else "WARNING",
    enqueue=True,
)

for val_txt in VAL_TXT:
    ap_table = "".join(os.popen(f"python3 /home/sun/Bev/BeMapNet/tools/evaluation/eval.py {gt_dir} {dt_dir} {val_txt}").readlines())
    logger.info(" AP-Performance with HDMapNetAPI: \n" + val_txt + "\n" + ap_table)





