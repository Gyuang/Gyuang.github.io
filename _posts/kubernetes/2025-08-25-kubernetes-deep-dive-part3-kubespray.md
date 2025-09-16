---
title: "[Kubernetes 심층 탐구 3부] Kubespray 실전 가이드"
date: 2025-08-25 09:00:00 +0900
categories:
  - kubernetes
tags:
  - kubernetes
  - kubespray
  - ansible
  - deployment
  - infrastructure-as-code
author_profile: true
sidebar:
  nav: "docs"
toc: true
toc_sticky: true
---

## 서론

안녕하세요. 'Kubernetes 심층 탐구' 시리즈의 세 번째 시간입니다. 1, 2부에서 쿠버네티스의 컨트롤 플레인과 워커 노드의 복잡한 내부 동작을 살펴보았습니다. 이번 포스트에서는 "이렇게 복잡한 시스템을 어떻게 하면 안정적이고 반복 가능하게 설치할 수 있을까?"라는 질문에 대한 해답, **Kubespray**에 대해 알아보겠습니다. 단순한 소개를 넘어, 실제 운영에 필요한 실전 팁과 고급 설정까지 다루는 가이드입니다.

## 1. 왜 Kubespray인가? Ansible 기반의 장점

Kubespray는 쿠버네티스 SIG(Special Interest Group)에서 관리하는 공식적인 클러스터 생명주기 관리 도구입니다. 핵심은 **Ansible**을 사용한다는 점입니다.

-   **Idempotency (멱등성)**: Ansible 플레이북은 '상태'를 정의합니다. 여러 번 실행해도 항상 정의된 상태와 동일한 결과가 보장됩니다. 예를 들어 "패키지 `nginx`를 설치하라"는 플레이북을 실행하면, `nginx`가 없으면 설치하고, 이미 있으면 아무것도 하지 않습니다. 이는 클러스터 설정 변경 및 업그레이드 시 예측 가능성과 안정성을 크게 높여줍니다.
-   **Agentless**: 대상 노드에 별도의 에이전트를 설치할 필요 없이, 제어 머신에서 SSH를 통해 모든 작업을 수행합니다. 이는 초기 구성을 매우 간편하게 만듭니다.
-   **Infrastructure as Code (IaC)**: 모든 클러스터 구성(노드 정보, 쿠버네티스 버전, 네트워크 설정 등)이 텍스트 파일(YAML, INI)로 관리됩니다. Git과 같은 버전 관리 시스템으로 클러스터의 변경 이력을 추적하고, 필요시 이전 상태로 롤백하는 것이 용이합니다.

## 2. Kubespray 디렉토리 구조 파헤치기

Kubespray를 제대로 활용하려면 디렉토리 구조를 이해해야 합니다.

```
kubespray/
├── inventory/              # 클러스터 구성 정보를 담는 곳
│   └── my-cluster/
│       ├── inventory.ini   # 노드 IP와 그룹 정보
│       └── group_vars/     # 그룹별 변수 정의
│           ├── all/all.yml
│           └── k8s_cluster/k8s-cluster.yml
├── roles/                  # Ansible 역할(ex: etcd, kubelet 설치) 정의
├── cluster.yml             # 클러스터 설치를 위한 메인 플레이북
├── upgrade-cluster.yml     # 클러스터 업그레이드 플레이북
├── scale.yml               # 노드 추가/삭제 플레이북
└── reset.yml               # 클러스터 초기화 플레이북
```

-   **`inventory/`**: 가장 중요한 디렉토리입니다. 클러스터의 '설계도'에 해당합니다. `inventory.ini` 파일에 노드들의 IP와 역할을 정의하고, `group_vars/` 디렉토리의 YAML 파일들에서 세부 설정을 변경합니다.
-   **`roles/`**: "etcd 설치", "kubelet 설정" 등 각 작업을 모듈화한 '역할'들이 모여있습니다. Kubespray의 실제 작업 내용은 대부분 이 `roles` 안에 정의되어 있습니다.
-   **`*.yml` (플레이북)**: `cluster.yml`과 같은 최상위 플레이북들은 `roles`에 정의된 역할들을 적절한 순서로 호출하여 전체 작업을 오케스트레이션합니다.

## 3. 실전! 고급 설정 및 커스터마이징

기본 설치를 넘어, 실제 운영 환경에 맞게 Kubespray를 커스터마이징하는 방법을 알아봅시다. 모든 설정은 `inventory/my-cluster/group_vars/` 안의 파일에서 이루어집니다.

### 고가용성(HA) 컨트롤 플레인 구성

`inventory.ini` 파일에 컨트롤 플레인 노드를 여러 개 지정하기만 하면 됩니다.

```ini
[kube_control_plane]
master1 ansible_host=192.168.1.11
master2 ansible_host=192.168.1.12
master3 ansible_host=192.168.1.13
```

Kubespray는 자동으로 `kube-apiserver` 앞에 로드 밸런서(Nginx 또는 HAProxy)를 설치하여 API 서버의 HA를 구성해 줍니다.

### CNI 플러그인 변경 및 설정

`k8s-cluster.yml` 파일에서 CNI 플러그인을 변경하고 세부 옵션을 조정할 수 있습니다.

```yaml
# inventory/my-cluster/group_vars/k8s_cluster/k8s-cluster.yml

# 사용할 CNI 플러그인 선택 (calico, cilium, flannel 등)
kube_network_plugin: cilium

# Calico를 사용할 경우, BGP 라우팅 모드 활성화
calico_network_backend: "bird"

# Cilium을 사용할 경우, eBPF 기반 kube-proxy 대체 기능 활성화
kube_proxy_mode: "cilium"
cilium_kube_proxy_replacement: "strict"
```

### 클러스터 애드온(Add-on) 관리

`addons.yml` 파일에서 다양한 애드온을 활성화/비활성화할 수 있습니다.

```yaml
# inventory/my-cluster/group_vars/k8s_cluster/addons.yml

# Kubernetes 대시보드 설치
dashboard_enabled: true

# 클러스터 오토스케일러 설치
cluster_autoscaler_enabled: true

# Prometheus 기반 모니터링 스택 설치
metrics_server_enabled: true
prometheus_enabled: true
grafana_enabled: true
```

## 4. 클러스터 생명주기 관리

Kubespray의 진가는 최초 설치 이후에 드러납니다.

### 노드 추가 (Scaling Up)

1.  새로운 노드의 IP를 `inventory.ini`의 `[all]`과 `[kube_node]` 그룹에 추가합니다.
2.  `scale.yml` 플레이북을 실행합니다.

```bash
ansible-playbook -i inventory/my-cluster/inventory.ini --become scale.yml
```

### 클러스터 업그레이드

1.  `k8s-cluster.yml` 파일에서 `kube_version` 변수를 원하는 버전으로 수정합니다.
    ```yaml
    kube_version: v1.29.0
    ```
2.  `upgrade-cluster.yml` 플레이북을 실행합니다.

```bash
ansible-playbook -i inventory/my-cluster/inventory.ini --become upgrade-cluster.yml
```
Kubespray는 컨트롤 플레인부터 워커 노드까지, 한 노드씩 순차적으로 드레이닝(draining)하고 업그레이드하여 서비스 중단 없는 롤링 업그레이드를 수행합니다.

### 트러블슈팅 팁

플레이북 실행 중 오류가 발생하면, Ansible의 출력 메시지를 잘 살펴보는 것이 중요합니다.

-   **어떤 작업(TASK)에서 실패했는가?**: 실패한 작업의 이름을 확인하여 어떤 역할을 수행하던 중이었는지 파악합니다.
-   **어떤 노드에서 실패했는가?**: 특정 노드에서만 문제가 발생하는지 확인합니다.
-   **오류 메시지(msg)는 무엇인가?**: "package not found", "permission denied" 등 구체적인 오류 메시지를 통해 원인을 유추할 수 있습니다.
-   **재시도**: 일시적인 네트워크 문제일 수 있으므로, `--limit @/path/to/failed_hosts.retry` 옵션을 사용하여 실패한 호스트에 대해서만 재시도할 수 있습니다.

## 결론

Kubespray는 복잡한 쿠버네티스 설치 및 관리 작업을 '코드로 관리되는 인프라'의 영역으로 가져오는 강력한 도구입니다. Ansible의 멱등성과 Kubespray의 정교한 역할(role) 설계를 통해, 우리는 버튼 하나로 프로덕션 레벨의 클러스터를 구축, 확장, 업그레이드할 수 있습니다. 수동 작업의 실수를 없애고, 누구든지 일관된 결과를 만들어낼 수 있다는 것이 Kubespray의 가장 큰 가치입니다.

이제 우리는 안정적으로 구축된 쿠버네티스 클러스터를 갖게 되었습니다. 마지막 4부에서는 이 클러스터 위에 AI/ML 워크로드를 위한 최적의 환경을 구축해주는 **DeepOps**에 대해 알아보겠습니다.
