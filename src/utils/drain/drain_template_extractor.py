# -*- coding: utf-8 -*-

"""
    抽取日志模板信息。
"""

import os
import sys
import drain3
import pandas as pd
from tqdm import tqdm

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)
import utils.io_util as io


def init_drain():
    config = TemplateMinerConfig()
    config_pth = os.path.join(
        os.path.dirname(__file__),
        "drain3.ini"
    )
    config.load(config_pth)
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    return template_miner


def extract_templates(log_list: list, save_pth: str):
    KEEP_TOP_N_TEMPLATE = 1000

    miner = init_drain()

    for line in tqdm(log_list):
        log_txt = line.rstrip()
        miner.add_log_message(log_txt)
    template_count = len(miner.drain.clusters)
    print('The number of templates: {}'.format(template_count))

    template_dict, size_list = {}, []
    for cluster in miner.drain.clusters:
        size_list.append(cluster.size)

    size_list = sorted(size_list, reverse=True)[:KEEP_TOP_N_TEMPLATE]
    min_size = size_list[-1]

    for c in miner.drain.clusters:
        if c.size >= min_size:
            template_dict[c.cluster_id] = c.size

    io.save(
        file=save_pth,
        data=miner
    )

    sorted_clusters = sorted(miner.drain.clusters, key=lambda it: it.size, reverse=True)
    print('[show templates]')
    for cluster in sorted_clusters:
        print(cluster)

    print("Prefix Tree:")
    miner.drain.print_tree()

    miner.profiler.report(0)

    return miner


def match_template(miner: drain3.TemplateMiner, log_list: list):
    # logger = get_logger("logs_matching")
    IDs = []
    templates = []
    params = []

    for log in tqdm(log_list):
        cluster = miner.match(log)
        
        # logger.debug('match log: {}'.format(log))
        if cluster is None:
            # logger.debug("No match found")
            IDs.append(None)
            templates.append(None)
            
        else:
            template = cluster.get_template()
            param = miner.get_parameter_list(template, log)

            IDs.append(cluster.cluster_id)
            templates.append(template)
            params.append(param)
            # logger.debug(f"Matched template #{cluster.cluster_id}: {template}")

            # params = miner.get_parameter_list(template, log)
            # logger.debug(f"Parameters: {params}")
        # logger.debug("===========================================================================================")

    return IDs, templates, params