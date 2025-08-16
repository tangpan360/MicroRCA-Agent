"""
存放各种prompt模板的模块
"""

def get_multimodal_analysis_prompt(
    log_data: tuple[str, dict] | None = None,
    trace_data: tuple[str, dict, str] | None = None,
    metric_data: str | None = None
) -> str:
    """
    获取多模态分析的prompt模板，支持缺失部分模态数据

    参数:
        log_data: (filtered_logs_csv, log_unique_dict) 或 None
        trace_data: (filtered_traces_csv, trace_unique_dict, status_combinations_csv) 或 None
        metric_data: 字符串类型的metric分析结果 或 None

    返回:
        构建好的多模态分析prompt字符串
    """

    # 固定的组件列表，不再从数据中动态提取
    all_node_names = ['aiops-k8s-01', 'aiops-k8s-02', 'aiops-k8s-03', 'aiops-k8s-04',
                      'aiops-k8s-05', 'aiops-k8s-06', 'aiops-k8s-07', 'aiops-k8s-08']

    all_service_names = ['cartservice', 'currencyservice', 'frontend', 'adservice',
                         'recommendationservice', 'shippingservice', 'checkoutservice',
                         'paymentservice', 'emailservice', 'redis-cart', 'productcatalogservice', 'tidb-tidb', 'tidb-pd', 'tidb-tikv']

    all_pod_names = ['cartservice-0', 'cartservice-1', 'cartservice-2', 'currencyservice-0',
                     'currencyservice-1', 'currencyservice-2', 'frontend-0', 'frontend-1',
                     'frontend-2', 'adservice-0', 'adservice-1', 'adservice-2',
                     'recommendationservice-0', 'recommendationservice-1', 'recommendationservice-2',
                     'shippingservice-0', 'shippingservice-1', 'shippingservice-2',
                     'checkoutservice-0', 'checkoutservice-1', 'checkoutservice-2',
                     'paymentservice-0', 'paymentservice-1', 'paymentservice-2',
                     'emailservice-0', 'emailservice-1', 'emailservice-2',
                     'productcatalogservice-0', 'productcatalogservice-1', 'productcatalogservice-2',
                     'redis-cart-0']

    available_modalities = []
    data_sections = []

    # 处理日志数据
    if log_data and log_data[0]:  # 检查是否有有效的CSV数据
        filtered_logs_csv, log_unique_dict = log_data
        available_modalities.append("日志数据")
        data_sections.append(f"""
### 日志异常数据:
包含列: node_name(节点名称), service_name(服务名称), pod_name(Pod名称), message(日志消息), occurrence_count(出现次数)
{filtered_logs_csv}
        """)

    # 处理trace数据
    if trace_data and trace_data[0]:  # 检查是否有有效的CSV数据
        filtered_traces_csv, trace_unique_dict, status_combinations_csv = trace_data
        available_modalities.append("链路追踪数据")

        # 添加孤立森林检测结果
        trace_section = f"""
### 微服务链路异常数据（孤立森林检测结果，前20个）:
包含列: node_name(节点名称), service_name(服务名称), parent_pod(父Pod), child_pod(子Pod), operation_name(父pod和子pod的调用操作), normal_avg_duration(正常平均耗时), anomaly_avg_duration(异常平均耗时), anomaly_count(异常次数)
{filtered_traces_csv}"""

        # 如果有status组合数据，也添加进去
        if status_combinations_csv:
            trace_section += f"""

### 微服务链路状态异常数据（status.code和status.message，前20个）:
包含列: node_name(节点名称), service_name(服务名称), parent_pod(父Pod), child_pod(子Pod), operation_name(操作名称), status_code(状态码), status_message(状态消息), occurrence_count(出现次数)
{status_combinations_csv}"""

        data_sections.append(trace_section)

    # 处理指标数据
    if metric_data:  # 检查是否有有效的字符串数据
        available_modalities.append("系统指标数据")
        data_sections.append(f"""
### 系统指标综合分析结果:
{metric_data}
        """)

    # # 如果没有任何有效数据，返回错误提示
    # if not data_sections:
    #     return "错误：未提供任何有效的监控数据，无法进行故障分析。"

    # 构建数据部分
    data_content = "\n".join(data_sections)

    # 构建包含三种类型组件的列表
    components_list = []
    components_list.extend(all_node_names)
    components_list.extend(all_service_names)
    components_list.extend(all_pod_names)
    modalities_text = "、".join(available_modalities)

    return f"""
        ### Language Enforcement
        -Input may contain Chinese, **but output MUST be entirely in English** (no Chinese characters).
        请根据提供的{modalities_text}，进行综合故障分析，识别最有可能的单一故障原因。不要包含任何其他解释或文本。
        特别注意**缺失数据和空数据,代表数据波动极小,通常情况默认正常,严禁分析和定位为根因**

        ### 微服务架构调用关系图谱
        理解以下关键调用路径有助于识别故障传播和根因定位：
        
        **主要调用路径:**
        1. **用户请求入口**: User → frontend (所有用户请求的统一入口)
        2. **购物核心流程**: frontend → checkoutservice → (paymentservice, emailservice, shippingservice, currencyservice)
        3. **商品浏览相关**: frontend → (adservice, recommendationservice, productcatalogservice, cartservice)
        4. **服务间依赖**: recommendationservice → productcatalogservice (推荐依赖商品目录)
        5. **数据存储层**:
           - adservice/productcatalogservice → tidb (广告和商品数据存储)
           - cartservice → redis-cart (购物车缓存)
           - tidb 集群内部: tidb → (tidb-tidb, tidb-tikv, tidb-pd)
           
        要求：
        1. 综合多种监控数据进行分析，优先考虑数据间的关联性
        2. 只返回一个最可能的故障分析结果
        3. 故障级别判断标准：
            **Node级别故障**: 单个节点的监控指标(kpi_key)（node_cpu_usage_rate,node_filesystem_usage_rate等）对比正常期间,故障期间存在显著异常变化，且该节点上的多个不同服务的Pod均受影响
            **Service级别故障**: 同一服务的多个Pod实例（如emailservice-0, emailservice-1, emailservice-2）都出现相似的异常数据变化，表明服务本身存在问题
            **Pod级别故障**: 单个Pod（如cartservice-0）出现异常数据变化，而同服务的其他Pod（cartservice-1, cartservice-2）及其他Pod正常
            **重要说明**：所有监控指标均为 `kpi_key` 指标（例如 `node_cpu_usage_rate`），请在描述中直接使用这些原始 `kpi_key` 英文指标名，不得使用中文或其他名称。
        4. 请确保:
            - component必须从提供的组件列表中选择，组件列表包含三种故障层级：
                * 节点名(aiops-k8s-01~08) - 表示节点级别的基础设施故障
                * 服务名(cartservice等) - 表示微服务级别的故障
                * Pod名(cartservice-0等) - 表示单个Pod级别的故障
            ### **Observation** and **Reason** Description Constraint
            - Both **observation** and **reason** fields must clearly mention the **kpi_key (metric name)** involved in the fault.  
            - Do not describe specific percentiles (Median,p50, interquartile range, IQR, 99th percentile, etc.) or any numeric values.  
            - Use only trend words to describe anomalies, e.g., `surged`, `dropped`, `spiked`, `declined`.  
            - Retain only the **kpi_key (metric name)** and the **component/service name, pod name, or node name** when describing anomalies.  
            - **Reason field**: Must specify which exact **kpi_key** is abnormal and briefly explain the root cause.  
            - **Observation field**: Must be based on multimodal evidence and explicitly indicate the source modality:  
            - If from **metric**, explicitly mention the abnormal **kpi_key**.  
            - If from **log**, mention the keyword(s) in logs.  
            - If from **trace**, describe the abnormal call behavior (caller/callee/self-loop) involving the fault component in the trace path. 
            - **特别要求**严禁分析和定位缺失数据和空数据为根因,默认其正常
        5. The JSON output must be fully in English. Any Chinese characters are strictly prohibited.
            **Strictly follow the JSON format below**：
        {{
            "component": "Select from the following components: {components_list}",
            "reason": "Most likely root cause based on comprehensive multi-modal analysis; (must include kpi_key for metrics. (Do not infer from missing data.))",
            "reasoning_trace": [
                {{
                    "step": 1,
                    "action": "Such as: LoadMetrics(checkoutservice)",
                    "observation": "Describe (≤20 words) the most critical anomaly in metric modality, must include exact kpi_key and change (e.g., '`node_cpu_usage_rate` increased 35% at 12:18 in metric')"
                }},
                {{
                    "step": 2,
                    "action": "Such as: TraceAnalysis('frontend-1 -> checkoutservice-2')", 
                    "observation": "Describe (≤20 words) the most critical abnormal behavior in trace modality, include trace path and anomaly type (caller/callee/self-loop) (e.g., 'self-loop detected in `frontend -> checkoutservice` in trace')"
                }},
                {{
                    "step": 3,
                    "action": "Such as: LogSearch(checkoutservice)",
                    "observation": "Describe (≤20 words) the most critical anomaly in log modality, mention error keyword and count/context (e.g., 'IOError found in 3 entries in log')"
                }}
            ]
        }}        

        可用的监控数据:
        {data_content}
        """
