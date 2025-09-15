# MicroRCA-Agent: LLM-Agent-Based Microservice Root Cause Analysis

English | [中文](README_zh.md)

## 2025 International AIOps Challenge (Finals Top 5, 48.52 points)

## Project Overview

This project is an intelligent operations solution based on multi-modal data analysis, capable of processing Log, Trace, and Metric data for fault analysis and root cause localization through large language models. It adopts a modular architecture design with five core modules: data preprocessing module, log fault extraction module, trace fault detection module, metric fault summarization module, and multi-modal root cause analysis module. The modules are designed with loose coupling through function encapsulation for data interaction, ensuring both system integrity and module independence and scalability.

The output contains structured root cause analysis results including component, reason, and reasoning_trace, achieving a complete closed loop from phenomenon observation to root cause reasoning.

![Project Architecture](imgs/overview.png)

## File Structure

```
├── README.md  # English project documentation
├── README_zh.md  # Chinese project documentation
├── domain.conf  # External domain configuration
├── src/  # Source code directory
│   ├── agent/  # Intelligent agent module
│   │   ├── __init__.py  # Package initialization file
│   │   ├── agents.py  # Agent implementation
│   │   ├── llm_config.py  # LLM configuration for agent model list
│   │   └── prompts.py  # Prompt templates
│   ├── utils/  # Utility modules
│   │   ├── drain/  # Drain log template extraction
│   │   │   ├── drain_template_extractor.py  # Drain template extractor
│   │   │   ├── drain3.ini  # Drain3 configuration file
│   │   │   ├── error_log-drain.pkl  # Pre-trained template extraction model
│   │   │   └── error_log-template.csv  # Log template file
│   │   ├── __init__.py  # Package initialization file
│   │   ├── file_utils.py  # File processing utilities
│   │   ├── io_util.py  # IO utilities
│   │   ├── llm_record_utils.py  # LLM record utilities
│   │   ├── log_template_extractor.py  # Log template extractor (for training error_log-drain.pkl)
│   │   ├── log_template_extractor_with_examples.py  # Log template extractor with examples
│   │   ├── log_utils.py  # Log processing utilities
│   │   ├── metric_utils.py  # Metric processing utilities
│   │   └── trace_utils.py  # Trace processing utilities
│   ├── models  # Models (trace anomaly detection models)
│   ├── scripts/  # Data preprocessing scripts including timestamp unification for log, trace, metric
│   │   ├── merge_phaseone_phasetwo_input_json.py  # Script to merge phaseone and phasetwo input.jsonl
│   │   ├── raw_log_processor.py  # Raw log processing
│   │   ├── raw_metric_processor.py  # Raw metric processing
│   │   └── raw_trace_processor.py  # Raw trace processing
│   ├── models/  # Model files
│   │   ├── trace_detectors.pkl  # Trace anomaly detection model
│   │   └── trace_detectors_normal_stats.pkl  # Trace normal state statistics
│   ├── input/  # Input data processing
│   │   ├── extract_input_timestamp.py  # Timestamp extraction
│   │   └── input_timestamp.csv  # Extracted input timestamp information
│   ├── submission/  # Submission results
│   │   ├── result.jsonl  # Result file
│   │   └── submit.py  # Submission script
│   ├── main_multiprocessing.py  # Main program entry
│   ├── preprocessing.sh  # Data preprocessing script
│   └── requirements.txt  # Python dependencies
├── data/  # Downloaded and preprocessed data files directory
├── Dockerfile  # Docker image build file
└── run.sh  # Startup script
```

## Technical Solution Details

### 1. Data Preprocessing

#### 1.1 Input Data Parsing Design
- **Regular Expression Extraction Strategy**: Structured processing of raw fault description files
  - Receives JSON format input data containing Anomaly Description and unique identifier (uuid)
  - Timestamp extraction mechanism: Uses ISO 8601 time format standard with regex pattern recognition for fault start/end times
  - Time index generation: Generates "year-month-day_hour" format time identifiers for quick data file location
  - Nanosecond-level timestamp conversion: Converts fault time to 19-digit nanosecond-level timestamps

#### 1.2 Multi-modal Data Timestamp Unification
- **Differentiated Timestamp Unification Strategy**: Targets different format characteristics of log, trace, and metric data
  - Log data: Uses ISO 8601 format @timestamp field, converts to unified 19-digit nanosecond-level timestamps
  - Trace data: startTime field stores microsecond-level timestamps, extends to nanosecond-level through precision conversion (multiply by 1000)
  - Metric data: time field follows ISO 8601 format, uses recursive search strategy to ensure complete coverage of distributed metric files
- **Temporal Consistency Guarantee**: After processing, sorts by timestamp in ascending order to ensure standardized cross-modal time baseline

### 2. Multi-modal Data Processing

#### 2.1 Log Data Processing
- **Drain3 Algorithm**: Trains Drain3 model based on error-containing logs and uses pre-trained Drain3 model (`error_log-drain.pkl`) for log template extraction
  - Automatically identifies log patterns, categorizes similar logs into the same template
  - Significantly reduces log data volume, extracts key error information
  - Used for log deduplication and frequency statistics
- **Multi-level Data Filtering Processing Pipeline**:
  - File location: Precisely matches log files within fault time windows based on time information
  - Time window filtering: Strict time boundary filtering based on nanosecond-level timestamps
  - Error keyword filtering: Extracts log entries containing error information, filters normal business logs
  - Core field extraction: Extracts key information such as time, container, node, and error messages
  - Fault template matching: Uses pre-trained Drain model for template matching and standardization
  - Sample deduplication statistics: Deduplicates repeated logs and counts frequencies to assess fault severity
  - Service information extraction: Maps Pod information to services, reconstructs into standardized format

#### 2.2 Trace Processing
- **Dual Anomaly Detection Strategy**: Combines performance and status dimensions to identify anomaly patterns in microservice call chains
  - Duration anomaly detection: Uses IsolationForest algorithm to detect call latency anomalies
  - Status detection: Directly checks status.code and status.message to identify error states
- **IsolationForest Performance Anomaly Detection**:
  - Pre-trained model storage: `trace_detectors.pkl` and `trace_detectors_normal_stats.pkl`
  - Trained on 40-minute normal period data after fault recovery, grouped by "parent_pod-child_pod-operation_name"
  - Uses 30-second sliding window to process duration features with 1% anomaly contamination rate
- **Status Code Direct Check Mechanism**:
  - Parses status.code and status.message fields in trace tags
  - Directly identifies anomalous status calls through conditional filtering (status.code≠0)
  - Provides deterministic error status identification and detailed error information
- **Call Relationship Mapping**: Extracts pod_name, service_name, node_name to establish complete call chain parent-child relationships
- **Structured Output**: Separately outputs top 20 duration anomalies and status anomaly combinations, including node, service, container, operation dimensions

#### 2.3 Metric Data Processing
- **Dual-level LLM Phenomenon Summarization Strategy**: Intelligent phenomenon identification and inductive analysis based on large language models
  - First level: Application performance monitoring phenomenon identification and summarization (APM metrics + TiDB database component metrics)
  - Second level: Infrastructure machine performance metrics comprehensive phenomenon summarization and correlation analysis
- **Multi-level Monitoring Metrics System**:
  - APM business monitoring: 7 core metrics (error_ratio, request, response, rrt, timeout, etc.)
  - Pod container level: 9 infrastructure metrics (cpu_usage, memory, network, filesystem, etc.)
  - Node level: 16 infrastructure metrics (cpu, memory, disk, network, TCP connections, etc.)
  - TiDB database: 3 components with 20 specialized metrics (query, duration, connection, raft, etc.)
- **Intelligent Data Filtering and Processing**:
  - Normal time period definition: Adjacent time windows before and after faults, from 10 minutes after previous fault end to current fault start, and from 10 minutes after current fault end to next fault start, avoiding fault "aftershock" effects
  - Statistical symmetry ratio filtering: Automatically filters stable metrics with change amplitude less than 5%
  - Outlier removal: Removes maximum and minimum 2 extreme values each to build stable statistical baseline
  - Pod-Service unified analysis: Automatically extracts Service identifiers through Pod names
- **LLM Summary Output Content**:
  - Application performance anomaly phenomena: Service-level overall trends and Pod-level individual differences
  - Infrastructure machine performance anomaly phenomena: Cross-container and node resource state changes
  - Cross-level correlation phenomenon patterns: Anomaly distribution characteristics and propagation path identification

### 3. High-Performance Processing

#### 3.1 Parallel Computing
- **Multi-process Architecture**: Dynamically adjusts process pool size based on CPU core count (default 0.5x core count)
- **Task Partitioning**: Partitions fault time periods for parallel processing

#### 3.2 Fault Tolerance Mechanism
- **Retry Strategy**: Maximum 3 retry attempts per time period
- **Exception Isolation**: Single time period processing failure doesn't affect overall pipeline
- **Data Missing Tolerance**: When certain types of data (log, trace, or metric) are missing, the system can continue analysis using available data

### 4. Root Cause Result Output Example:
```bash
{
  "uuid": "33c11d00-2",
  "component": "checkoutservice",
  "reason": "disk IO overload",
  "reasoning_trace": [
    {
      "step": 1,
      "action": "LoadMetrics(checkoutservice)",
      "observation": "disk_read_latency spike"
    },
    {
      "step": 2,
      "action": "TraceAnalysis('frontend -> checkoutservice')",
      "observation": "checkoutservice self-loop spans"
    },
    {
      "step": 3,
      "action": "LogSearch(checkoutservice)",
      "observation": "IOError in 3 logs"
    }
  ]
}
```

## Prerequisites Installation

### Git LFS Installation (Required)

Since the project dataset and weight files are managed using Git LFS, you need to install and configure Git LFS before running.

#### 🐧 Installing Git LFS on Ubuntu

**✅ Step 1: Add Git LFS Repository**
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
```
This command automatically adds the official Git LFS APT source.

**✅ Step 2: Install Git LFS**
```bash
sudo apt-get install git-lfs
```

**✅ Step 3: Initialize Git LFS**
After installation, run the following command to enable Git LFS:
```bash
git lfs install
```
This will configure Git to support LFS functionality.

**🔍 Verify Git LFS Installation**
Verify that Git LFS is correctly installed and enabled:
```bash
git lfs version
```
Example output:
```
git-lfs/3.6.1 (GitHub; linux amd64; go 1.23.3)
```

### Python Dependencies Installation (Required)

The project's data preprocessing scripts (Python scripts called in `src/preprocessing.sh`) require dependencies specified in `src/requirements.txt`.

**✅ Create conda environment and install dependencies**

```bash
# Create Python 3.10 environment
conda create -n microrca python=3.10

# Activate environment
conda activate microrca

# Enter project directory and install dependencies
cd MicroRCA-Agent
pip install -r src/requirements.txt
```

**⚠️ Important Notes**
- Dependencies must be installed before preprocessing stage, otherwise data processing scripts will fail
- Other dependency installation methods (such as virtualenv, pipenv, etc.) can be configured according to personal preferences

## Configuration

### Environment Variables

Please add your own DeepSeek official API keys by configuring the following environment variables in the `src/.env` file:

- `KEJIYUN_API_KEY`: LLM API key
- `KEJIYUN_API_BASE`: LLM API base address

### Model Configuration

You need to set the models to use in `src/agent/llm_config.py`. The default enabled model is `deepseek-chat`. If you need to use other models, please add them yourself.

## Usage

### Quick Start

**⚠️ Important Reminder: Make sure you have completed the above configuration before running!**

```bash
bash run.sh
```

## Evaluation Submission and Result Query

### 🏆 Evaluation Platform
This project participates in the **CCF AIOps 2025 Challenge** in the "LLM-Agent-Based Microservice Root Cause Analysis" track.

**Official Evaluation Platform**: [https://challenge.aiops.cn/home/competition/1963605668447416345](https://challenge.aiops.cn/home/competition/1963605668447416345)

### 📋 Evaluation Process

#### **Step 1: Platform Registration and Login**
1. Visit the [evaluation platform](https://challenge.aiops.cn/home/competition/1963605668447416345) and complete account registration
2. After login, enter the "LLM-Agent-Based Microservice Root Cause Analysis" competition page

#### **Step 2: Obtain Team Identifier**
1. After successful login, click the "**Team**" tab at the top of the page
2. Find and copy your "**Team ID**" from the team information page

#### **Step 3: Configure Submission Credentials**
Edit the `submission/submit.py` file, find the `TICKET` variable and replace it with your team ID:
```python
TICKET = "your_team_id_here"  # Please replace this with your actual team ID
```

#### **Step 4: Run Algorithm to Generate Prediction Results**
Execute the main program in the project root directory:
```bash
bash run.sh
```
> ⏱️ After the program completes, a `result.jsonl` file will be automatically generated in the project root directory

#### **Step 5: Prepare Submission File**
Move the generated result file to the submission directory:
```bash
cp result.jsonl submission/result.jsonl
```

#### **Step 6: Execute Result Submission**
Switch to the submission directory and run the submission script:
```bash
cd submission
python submit.py
```

**✅ Successful Submission Example**:
```
Success! Your submission ID is 1757931486600
```
> 📝 Please record this submission ID for subsequent score queries

#### **Step 7: Query Evaluation Score**
Use the obtained submission ID to query the score:
```bash
python submit.py -i 1757931486600
```

**📊 Score Result Example**:
```
Submission 1757931486600 score: 0.3275
```
> 🎯 The closer the score is to 1.0, the higher the prediction accuracy

## Troubleshooting

### 1. Docker Related Issues

**Problem**: Docker service not running
```
Error: Docker service not running or insufficient permissions
```

**Solution**:
```bash
# Start Docker service
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER
# Re-login or execute
newgrp docker
```

**Problem**: Docker image build failure
```
Error: Docker image build failed
```

**Solution**:

1. Configure accelerator

```bash
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com",
    "https://docker.mirrors.ustc.edu.cn",
    "https://docker.nju.edu.cn",
    "https://vp5v3vra.mirror.aliyuncs.com",
    "https://docker.registry.cyou",
    "https://docker-cf.registry.cyou",
    "https://dockercf.jsdelivr.fyi",
    "https://docker.jsdelivr.fyi",
    "https://dockertest.jsdelivr.fyi",
    "https://mirror.baidubce.com",
    "https://docker.m.daocloud.io",
    "https://docker.nju.edu.cn",
    "https://docker.mirrors.sjtug.sjtu.edu.cn",
    "https://docker.mirrors.ustc.edu.cn",
    "https://mirror.iscas.ac.cn",
    "https://docker.rainbond.cc"
  ]
}
EOF
```

2. Restart Docker service

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### 2. Network Connection Issues

**Problem**: LLM API access failure

**Solution**:
- Check if environment variables `KEJIYUN_API_KEY` and `KEJIYUN_API_BASE` in the `src/.env` file are correctly configured

### 3. Memory Shortage Issues

**Problem**: Container crashes due to insufficient memory, system freeze

**Solution**:
- You can manually modify the number of processes. Adjust the process pool size in `src/main_multiprocessing.py` (default uses 50% of core count). However, if memory is too small, it may cause overflow and system freeze. Please monitor memory usage. If it overflows, please modify to an appropriate ratio:
```python
num_processes = max(1, int(cpu_count() * 0.5))
```

## Important Notes

1. Ensure all dependent external services (LLM API) are accessible
2. Recommend running on high-performance machines, as processing large amounts of data may take considerable time

## Acknowledgments

Thanks to the CCF AIOps 2025 Challenge organizing committee for providing high-quality datasets and a good competition environment, offering our team a valuable learning and exchange platform.

This project participated in: **Track 1: LLM-Agent-Based Microservice Root Cause Analysis**  
Competition website: [CCF AIOps 2025 Challenge](https://challenge.aiops.cn/home/competition/1920410697896845344)

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=tangpan360/MicroRCA-Agent&type=Date)
