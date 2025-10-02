---
title: '[심층 분석] 쿠버네티스, Kubespray, DeepOps: 아키텍처부터 MLOps 플랫폼 구축까지'
date: 2025-08-22 11:00:00 +0900
categories:
- server
- kubernetes
tags:
- kubernetes
- kubespray
- deepops
- infrastructure
- MLOps
- ansible
- gpu
author_profile: true
sidebar:
  nav: docs
toc: true
toc_sticky: true
---

안녕하세요! 이번 포스트에서는 컨테이너 오케스트레이션의 표준, **쿠버네티스(Kubernetes)**의 아키텍처를 심도 있게 분석하고, 실제 프로덕션 환경에서 클러스터를 구축하는 강력한 도구인 **Kubespray**와 **DeepOps**에 대해 상세히 알아보겠습니다. 이 글을 통해 단순히 개념을 이해하는 것을 넘어, 각 기술이 어떻게 유기적으로 연결되어 최종적인 MLOps 플랫폼을 구성하는지 파악하는 것을 목표로 합니다.

## 1. 쿠버네티스(Kubernetes) 아키텍처 심층 분석

쿠버네티스는 여러 서버(노드)를 하나의 거대한 컴퓨팅 리소스로 묶어주는 시스템입니다. 이 시스템은 클러스터를 지휘하는 **컨트롤 플레인**과 실제 작업이 이루어지는 **워커 노드**로 나뉩니다.

![Kubernetes Architecture](https://user-images.githubusercontent.com/16943343/200911239-93a3658a-7459-4345-9a98-4532a47791c3.png)
*<center>쿠버네티스 아키텍처 (출처: 쿠버네티스 공식 문서)</center>*

### 컨트롤 플레인 (Control Plane): 클러스터의 두뇌

컨트롤 플레인은 클러스터의 모든 것을 결정하고 관리하는 중앙 지휘 본부입니다. 고가용성(HA)을 위해 보통 3대 이상의 홀수 개 노드로 구성합니다.

-   **`kube-apiserver` (API 서버)**
    -   **역할**: 클러스터의 유일한 관문(Gateway)입니다. `kubectl` CLI, UI 대시보드, 다른 컴포넌트 등 모든 요청은 이 API 서버를 통해서만 클러스터와 상호작용할 수 있습니다. RESTful API 형태로 클러스터의 모든 조작을 지원합니다.
    -   **주요 기능**: 요청에 대한 인증(Authentication), 인가(Authorization, e.g., RBAC), 어드미션 컨트롤(Admission Control)을 수행하여 클러스터의 보안과 안정성을 책임집니다. 상태 비저장(Stateless)으로 설계되어 수평 확장이 용이합니다.

-   **`etcd` (분산 저장소)**
    -   **역할**: 클러스터의 모든 설정 값, 상태 데이터, 객체 명세(Spec) 등을 저장하는 분산 key-value 저장소입니다. Raft 합의 알고리즘을 사용하여 데이터의 일관성과 고가용성을 보장합니다.
    -   **중요성**: `etcd`가 손상되면 클러스터의 상태를 복구할 수 없으므로, 클러스터에서 가장 중요한 컴포넌트입니다. 따라서 주기적인 백업이 필수적이며, 보통 컨트롤 플레인 노드에 함께 배치하거나 별도의 전용 노드에 구성합니다.

-   **`kube-scheduler` (스케줄러)**
    -   **역할**: 새로 생성된 파드(Pod)를 어떤 워커 노드에 배치할지 결정합니다.
    -   **동작 방식 (2단계)**:
        1.  **필터링(Filtering)**: 파드가 요구하는 리소스(CPU, memory), 노드 셀렉터, 어피니티/안티-어피니티 규칙, Persistent Volume 요구사항 등을 만족하는 노드들의 목록을 추립니다.
        2.  **스코어링(Scoring)**: 필터링된 노드들을 대상으로 점수를 매깁니다. 리소스가 여유로운 노드, 필요한 이미지가 이미 캐시된 노드 등에 높은 점수를 부여하여 최적의 노드를 선택합니다.

-   **`kube-controller-manager` (컨트롤러 관리자)**
    -   **역할**: 클러스터의 상태를 "의도한 상태(desired state)"로 유지하기 위해 다양한 컨트롤러를 실행하는 데몬입니다. 각 컨트롤러는 **조정 루프(reconciliation loop)**를 통해 끊임없이 현재 상태를 감시하고 차이가 발생하면 조치를 취합니다.
    -   **주요 컨트롤러**:
        -   `Node Controller`: 노드의 상태를 주기적으로 확인하고, 응답이 없을 경우 해당 노드를 `NotReady` 상태로 변경합니다.
        -   `Replication Controller`: ReplicaSet 객체에 정의된 파드 개수가 실제 실행 중인 파드 개수와 동일하게 유지되도록 관리합니다.
        -   `Deployment Controller`: 배포(Deployment)의 롤링 업데이트와 같은 복잡한 시나리오를 관리합니다.

### 워커 노드 (Worker Node): 실제 작업 공간

워커 노드는 애플리케이션 컨테이너(파드)가 실제로 생성되고 실행되는 곳입니다.

-   **`kubelet` (노드 에이전트)**
    -   **역할**: 각 워커 노드에서 실행되는 핵심 에이전트입니다. API 서버로부터 자신의 노드에 할당된 파드 목록(PodSpec)을 받아와, 컨테이너 런타임에 전달하여 컨테이너를 실행하고 관리합니다.
    -   **주요 기능**: 컨테이너의 상태를 주기적으로 점검(Liveness/Readiness Probes)하고, 그 결과를 API 서버에 보고하여 노드와 파드의 상태를 업데이트합니다.

-   **`kube-proxy` (네트워크 프록시)**
    -   **역할**: 클러스터의 서비스(Service) 네트워킹을 담당합니다. 특정 서비스로 향하는 가상 IP(ClusterIP) 트래픽을 실제 파드의 IP로 전달하는 네트워크 규칙을 각 노드에 설정합니다.
    -   **동작 모드**: `iptables` (전통적이고 안정적), `IPVS` (대규모 클러스터에서 더 나은 성능) 등의 모드로 동작하며, 노드 내에서 로드 밸런싱과 서비스 디스커버리를 가능하게 합니다.

-   **`컨테이너 런타임` (Container Runtime)**
    -   **역할**: 컨테이너의 생명주기를 관리하는 소프트웨어입니다. `kubelet`은 CRI(Container Runtime Interface)라는 표준화된 인터페이스를 통해 런타임과 통신합니다.
    -   **종류**: `containerd` (현재 de-facto 표준), `CRI-O` 등이 있으며, 과거에는 Docker가 많이 사용되었습니다.

---

## 2. Kubespray: 프로덕션 레벨 클러스터 자동 구축

**Kubespray**는 Ansible을 기반으로 쿠버네티스 클러스터의 설치 및 수명주기 관리를 자동화하는 오픈소스 프로젝트입니다. 수동으로 클러스터를 구축할 때 발생하는 복잡성과 실수를 줄여주고, 일관성 있고 반복 가능한 배포를 가능하게 합니다.

### Kubespray의 장점과 특징

-   **Ansible 기반**: 에이전트가 필요 없는(agentless) 방식으로 작동하며, SSH만으로 타겟 노드를 설정합니다.
-   **높은 커스터마이징**: 쿠버네티스 버전, CNI(네트워크 플러그인), 컨테이너 런타임 등 거의 모든 요소를 변수 파일을 통해 손쉽게 변경할 수 있습니다.
-   **프로덕션 레디**: 고가용성(HA) 구성, 보안 설정(Hardening), 인증서 관리 등을 기본적으로 지원합니다.

### Kubespray 배포 상세 과정

1.  **사전 준비**: Ansible을 실행할 제어 머신에 `python`, `pip`, `git`을 설치합니다.
2.  **Kubespray 클론 및 의존성 설치**:
    ```bash
    git clone https://github.com/kubernetes-sigs/kubespray.git
    cd kubespray
    pip install -r requirements.txt
    ```
3.  **인벤토리(Inventory) 구성**:
    -   샘플 인벤토리를 복사하여 자신만의 클러스터 구성을 정의합니다.
        ```bash
        cp -rfp inventory/sample inventory/my-cluster
        ```
    -   `inventory/my-cluster/inventory.ini` 파일을 열어 노드 정보를 입력합니다.
        ```ini
        [all]
        node1 ansible_host=192.168.1.11 ip=192.168.1.11
        node2 ansible_host=192.168.1.12 ip=192.168.1.12
        node3 ansible_host=192.168.1.13 ip=192.168.1.13

        [kube_control_plane]
        node1

        [etcd]
        node1

        [kube_node]
        node2
        node3

        [k8s_cluster:children]
        kube_control_plane
        kube_node
        ```
4.  **설정 변수 커스터마이징**:
    -   `inventory/my-cluster/group_vars/k8s_cluster/k8s-cluster.yml`: 클러스터 버전, 네트워크 플러그인 등 전역 설정을 변경합니다.
        ```yaml
        kube_version: v1.28.5
        kube_network_plugin: calico
        container_manager: containerd
        ```
    -   `inventory/my-cluster/group_vars/all/all.yml`: SSH 사용자, 키 파일 등 Ansible 연결 정보를 설정합니다.

5.  **Ansible 플레이북 실행**:
    ```bash
    ansible-playbook -i inventory/my-cluster/inventory.ini --become --become-user=root cluster.yml
    ```
    -   `--become`: `sudo`와 같이 권한을 상승시켜 작업을 수행하도록 합니다. 이 명령 하나로 수십, 수백 개의 노드에 일관된 쿠버네티스 클러스터가 설치됩니다.

---

## 3. DeepOps: MLOps를 위한 올인원 플랫폼

**DeepOps**는 NVIDIA에서 제공하는 HPC 및 MLOps 환경 구축 자동화 도구셋입니다. DeepOps의 핵심은 **Kubespray를 기반으로 쿠버네티스를 설치**한 뒤, GPU를 활용하는 AI/ML 워크로드에 필수적인 구성 요소들을 그 위에 추가로 배포하는 것입니다.

### DeepOps가 Kubespray 위에 추가하는 것들

DeepOps는 단순한 K8s 클러스터를 넘어, 데이터 사이언티스트와 ML 엔지니어가 즉시 사용할 수 있는 완전한 플랫폼을 제공합니다.

-   **NVIDIA GPU Operator**:
    -   **역할**: 쿠버네티스에서 NVIDIA GPU를 완벽하게 지원하기 위한 모든 것을 자동화합니다.
    -   **설치 요소**: NVIDIA 드라이버, Kubernetes용 NVIDIA 컨테이너 툴킷, DCGM(GPU 모니터링), 노드 피처 디스커버리(NFD) 등을 포함한 오퍼레이터입니다.
    -   **효과**: 관리자는 복잡한 드라이버 설치 과정 없이, 파드 명세에 `resources: { limits: { nvidia.com/gpu: 1 } }` 와 같이 간단히 선언하는 것만으로 GPU를 할당할 수 있습니다.

-   **Slurm on Kubernetes (선택 사항)**:
    -   **역할**: 전통적인 HPC 환경에서 널리 쓰이는 워크로드 매니저 `Slurm`을 쿠버네티스 위에서 실행할 수 있도록 통합합니다.
    -   **효과**: 기존 Slurm 스크립트(`sbatch`)에 익숙한 연구자들이 환경 변화 없이 컨테이너 기반의 워크로드를 실행할 수 있게 하여, HPC와 클라우드 네이티브 생태계의 간극을 메웁니다.

-   **모니터링 및 로깅 스택**:
    -   **Prometheus**: 클러스터의 모든 컴포넌트와 `DCGM-exporter`를 통해 수집된 GPU 메트릭(사용률, 온도, 메모리 등)을 수집합니다.
    -   **Grafana**: Prometheus가 수집한 데이터를 시각화하는 대시보드를 제공합니다. DeepOps는 GPU 사용 현황, 클러스터 자원 현황 등을 바로 확인할 수 있는 사전 구성된 대시보드를 포함합니다.
    -   **Loki/Promtail**: 클러스터 전반의 로그를 수집하고 검색할 수 있는 로깅 시스템을 구축합니다.

-   **스토리지 및 데이터 관리**:
    -   **NFS/Ceph**: 대용량 데이터셋과 모델 체크포인트를 저장하고 여러 파드에서 공유할 수 있도록 공유 스토리지 솔루션을 쉽게 연동할 수 있는 플레이북을 제공합니다.

-   **JupyterHub**:
    -   **역할**: 멀티-유저 Jupyter 노트북 환경을 제공하여, 각 사용자가 격리된 환경에서 GPU 자원을 할당받아 대화형으로 코드를 실행하고 실험할 수 있도록 지원합니다.

## 결론: 계층적 관점으로 본 세 기술의 관계

> **DeepOps (The AI Platform)**
> 
> *GPU Operator, Slurm, Monitoring, Storage...*
> 
> ---
> 
> **Kubespray (The Automated Installer)**
> 
> *Ansible Playbooks for K8s Deployment*
> 
> ---
> 
> **Kubernetes (The Core System)**
> 
> *Control Plane, Worker Nodes, API...*

-   **쿠버네티스**는 컨테이너화된 애플리케이션을 조율하기 위한 강력하고 유연한 **핵심 시스템**입니다.
-   **Kubespray**는 이 복잡한 시스템을 자동화된 방식으로, 일관성 있게, 그리고 프로덕션 환경에 맞게 구축해주는 **전문 설치 도구**입니다.
-   **DeepOps**는 Kubespray를 활용하여 기반을 다진 후, 그 위에 AI/ML 및 HPC 워크로드에 특화된 전문 장비(GPU 지원, 모니터링, 스케줄러 등)를 얹어 완성된 **목적 지향적 플랫폼**을 구축하는 솔루션입니다.

따라서 MLOps 또는 HPC를 위한 인프라를 구축하고자 할 때, 이 세 가지 기술의 관계와 각자의 역할을 깊이 이해하는 것은 성공적인 시스템 설계와 운영의 첫걸음이 될 것입니다.
