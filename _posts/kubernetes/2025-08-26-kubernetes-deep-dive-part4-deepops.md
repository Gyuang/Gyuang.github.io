---
title: "[Kubernetes 심층 탐구 4부] DeepOps로 MLOps 플랫폼 구축하기"
date: 2025-08-26 09:00:00 +0900
categories:
  - kubernetes
tags:
  - kubernetes
  - deepops
  - MLOps
  - gpu-operator
  - slurm
  - monitoring
author_profile: true
sidebar:
  nav: "docs"
toc: true
toc_sticky: true
---

## 서론

안녕하세요. 'Kubernetes 심층 탐구' 시리즈의 대망의 마지막 편입니다. 우리는 지금까지 쿠버네티스의 내부 아키텍처를 배우고(1, 2부), Kubespray를 통해 클러스터를 안정적으로 구축하는 방법(3부)까지 알아보았습니다. 이번 시간에는 이 모든 것을 기반으로, 최종 목표인 **AI/ML 워크로드를 위한 MLOps 플랫폼**을 완성시켜주는 **DeepOps**에 대해 심층적으로 분석해 보겠습니다.

## 1. DeepOps란? Kubespray를 품은 MLOps 플랫폼

DeepOps는 NVIDIA에서 제공하는 오픈소스 프로젝트로, 한마디로 **"MLOps 및 HPC 환경 구축을 위한 자동화된 레시피 모음"**입니다. DeepOps의 가장 큰 특징은 밑바닥부터 모든 것을 새로 만드는 것이 아니라, 이미 검증된 최고의 오픈소스 도구들을 조합하여 최적의 플랫폼을 구축한다는 점입니다.

그 중심에는 **Kubespray**가 있습니다. DeepOps의 설치 과정은 다음과 같이 요약할 수 있습니다.

> 1.  **1단계 (기반 공사)**: **Kubespray**를 사용하여 견고한 쿠버네티스 클러스터를 설치합니다.
> 2.  **2단계 (전문 설비 추가)**: 그 위에 GPU 지원, 고성능 스케줄러, 모니터링, 스토리지 등 MLOps에 특화된 컴포넌트들을 **Ansible과 Helm 차트**를 이용해 추가로 배포합니다.

즉, DeepOps는 Kubespray의 확장팩이자, AI/ML이라는 특정 목적에 맞게 고도로 튜닝된 솔루션입니다.

## 2. DeepOps의 핵심 컴포넌트 심층 분석

DeepOps가 제공하는 가치는 Kubespray가 설치한 기본 클러스터 위에 추가하는 컴포넌트들에 있습니다.

### NVIDIA GPU Operator: K8s에서 GPU를 손쉽게

쿠버네티스에서 GPU를 사용하려면 노드에 드라이버 설치, 컨테이너 툴킷 설정 등 복잡한 과정이 필요합니다. **GPU Operator**는 이 모든 것을 자동화합니다.

-   **동작 원리**: Operator 패턴을 사용하여 GPU 관련 컴포넌트의 생명주기를 관리합니다.
    1.  **Node Feature Discovery (NFD)**: 클러스터의 노드들을 스캔하여 GPU가 장착된 노드를 찾아내고 `nvidia.com/gpu=true`와 같은 레이블을 자동으로 붙여줍니다.
    2.  **컴포넌트 배포**: GPU가 있는 노드에만 DaemonSet 형태로 다음 컴포넌트들을 배포합니다.
        -   **NVIDIA 드라이버**: 호스트 OS를 건드리지 않고, 컨테이너화된 드라이버를 설치하여 커널 모듈을 로드합니다.
        -   **NVIDIA Container Toolkit**: 컨테이너가 GPU에 접근할 수 있도록 런타임을 설정합니다.
        -   **DCGM Exporter**: GPU의 상세한 메트릭(사용률, 메모리, 온도, 전력 등)을 Prometheus가 수집할 수 있도록 노출합니다.
-   **사용자 경험**: 개발자는 복잡한 설정 없이, 파드 명세에 다음과 같이 필요한 GPU 개수만 선언하면 됩니다.
    ```yaml
    spec:
      containers:
      - name: cuda-container
        image: nvidia/cuda:12.1.0-base-ubuntu22.04
        resources:
          limits:
            nvidia.com/gpu: 1 # GPU 1개를 요청
    ```

### Slurm on Kubernetes: HPC와 클라우드 네이티브의 만남

많은 연구 기관과 대학에서는 여전히 전통적인 HPC 워크로드 스케줄러인 **Slurm**을 사용합니다. DeepOps는 Slurm을 쿠버네티스 위에서 실행할 수 있도록 통합하여, 기존 사용자들의 경험을 유지하면서 컨테이너의 이점을 누릴 수 있게 합니다.

-   **통합 방식**: `slurm-operator`가 쿠버네티스에 Slurm 클러스터(slurmctld, slurmd, slurmdbd)를 파드 형태로 배포합니다. 사용자가 `sbatch` 명령으로 작업을 제출하면, Slurm은 `Pyxis`와 같은 도구를 통해 실제 작업을 컨테이너로 실행하고 쿠버네티스가 이를 스케줄링합니다.

### 모니터링 스택: 클러스터와 GPU 성능의 시각화

DeepOps는 **Prometheus**와 **Grafana**를 기본으로 완벽한 모니터링 환경을 제공합니다.

-   **Prometheus**: `kube-state-metrics`, `node-exporter` 등을 통해 클러스터의 전반적인 상태를 수집하고, 특히 GPU Operator의 `dcgm-exporter`를 통해 GPU의 상세한 성능 지표를 수집합니다.
-   **Grafana**: 수집된 데이터를 시각화합니다. DeepOps는 NVIDIA에서 미리 만들어 놓은 **GPU 대시보드**를 기본으로 제공하여, 클러스터 전체의 GPU 사용 현황, 개별 GPU의 부하, 메모리 사용량, 온도 등을 한눈에 파악할 수 있게 해줍니다. 이는 GPU 자원의 효율적인 활용과 문제 해결에 필수적입니다.

## 3. DeepOps 실전 가이드

DeepOps의 배포 과정은 Kubespray와 유사하지만, 자체적인 설정 파일을 사용합니다.

1.  **DeepOps 클론 및 설정 파일 준비**:
    ```bash
    git clone https://github.com/NVIDIA/deepops.git
    cd deepops
    cp config.example.yml config.yml
    ```
2.  **`config.yml` 설정**: 이 파일이 DeepOps의 인벤토리이자 변수 파일입니다.
    ```yaml
    # config.yml
    # Ansible 연결 정보
    ansible_user: ubuntu
    ansible_ssh_private_key_file: "~/.ssh/id_rsa"

    # 클러스터 노드 정보
    k8s_cluster_nodes:
      - { ip: "192.168.1.11", role: ["control-plane", "etcd", "schedulable"] }
      - { ip: "192.168.1.12", role: ["gpu-worker", "schedulable"] }
      - { ip: "192.168.1.13", role: ["cpu-worker", "schedulable"] }

    # 설치할 컴포넌트 활성화
    deepops_services:
      gpu-operator: { enabled: true }
      k8s-dashboard: { enabled: true }
      monitoring_stack: { enabled: true }
      slurm: { enabled: false }
    ```
3.  **Ansible 플레이북 실행**:
    ```bash
    ansible-playbook -l k8s-cluster playbooks/k8s-cluster.yml
    ```
    DeepOps는 목적에 따라 다양한 플레이북을 제공합니다. 위 명령은 쿠버네티스 클러스터와 `config.yml`에 활성화된 서비스들을 설치합니다.

## 결론: MLOps를 위한 최종 진화

'Kubernetes 심층 탐구' 시리즈를 통해 우리는 쿠버네티스라는 강력한 기반 위에 Kubespray로 안정적인 건물을 짓고, 마지막으로 DeepOps를 통해 그 건물을 최첨단 AI 연구소로 리모델링하는 전 과정을 살펴보았습니다.

-   **Kubernetes**: 유연하고 확장 가능한 **운영체제**
-   **Kubespray**: 그 운영체제를 설치하는 **자동화된 설치 마법사**
-   **DeepOps**: AI/ML이라는 특정 목적에 맞게 모든 것을 갖춘 **턴키(Turn-key) 솔루션**

DeepOps를 활용함으로써, 인프라팀은 복잡한 MLOps 플랫폼 구축 과정을 몇 시간 안에 완료할 수 있고, 데이터 사이언티스트와 ML 엔지니어들은 인프라 걱정 없이 즉시 연구와 개발에 집중할 수 있는 환경을 제공받게 됩니다. 이것이 바로 DeepOps가 제공하는 궁극적인 가치입니다.
