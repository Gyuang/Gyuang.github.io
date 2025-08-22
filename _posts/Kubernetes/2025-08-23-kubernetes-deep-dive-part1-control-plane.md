---
title: "[Kubernetes 심층 탐구 1부] 컨트롤 플레인 완전 정복"
date: 2025-08-23 09:00:00 +0900
categories:
  - kubernetes
tags:
  - kubernetes
  - control-plane
  - apiserver
  - etcd
  - scheduler
  - controller-manager
author_profile: true
sidebar:
  nav: "docs"
toc: true
toc_sticky: true
---

## 서론

안녕하세요. 'Kubernetes 심층 탐구' 시리즈의 첫 번째 포스트에 오신 것을 환영합니다. 이번 시간에는 쿠버네티스 클러스터의 두뇌, **컨트롤 플레인(Control Plane)**을 구성하는 각 컴포넌트의 역할과 내부 동작 방식을 아주 상세하게 파헤쳐 보겠습니다. 컨트롤 플레인을 이해하는 것은 쿠버네티스 전체를 이해하는 것과 같습니다.

## 1. `kube-apiserver`: 클러스터의 유일한 관문

`kube-apiserver`는 클러스터의 모든 상호작용이 거쳐 가는 중앙 허브입니다. 단순히 API 요청을 처리하는 것을 넘어, 클러스터의 보안과 일관성을 책임지는 핵심적인 역할을 수행합니다.

### API 요청 처리의 3단계

`kube-apiserver`에 들어온 요청은 크게 세 단계를 거쳐 처리됩니다.

1.  **인증 (Authentication)**: 요청을 보낸 사용자가 누구인지 식별하는 과정입니다. 쿠버네티스는 다양한 인증 방식을 지원합니다.
    -   **X.509 인증서**: `kubelet`이나 관리자가 사용하는 `kubectl` 설정 파일(`kubeconfig`)에 포함된 인증서로 사용자를 식별합니다. 가장 일반적인 방식입니다.
    -   **서비스 어카운트 토큰**: 클러스터 내부의 파드가 API 서버에 접근할 때 사용하는 Bearer 토큰입니다.
    -   **OpenID Connect (OIDC)**: 외부 인증 시스템(Google, Okta 등)과 연동할 때 사용됩니다.

2.  **인가 (Authorization)**: 인증된 사용자가 요청한 작업을 수행할 권한이 있는지 확인하는 과정입니다.
    -   **RBAC (Role-Based Access Control)**: "누가(Subject), 무엇을(Verb), 어떤 리소스에(Resource)" 할 수 있는지 정의하는 방식입니다. `Role`/`ClusterRole`로 권한을 정의하고, `RoleBinding`/`ClusterRoleBinding`으로 사용자와 역할을 연결합니다. 현재 사실상의 표준입니다.
    -   **ABAC (Attribute-Based Access Control)**: 속성 기반으로 더 복잡한 규칙을 만들 수 있지만, 관리가 어려워 잘 사용되지 않습니다.

3.  **어드미션 컨트롤 (Admission Control)**: 인가까지 통과한 요청이 클러스터에 최종적으로 적용되기 전, 추가적인 검증 및 수정을 가하는 단계입니다. 여러 개의 어드미션 컨트롤러가 체인 형태로 동작합니다.
    -   **Mutating Admission Webhooks**: 리소스 객체를 **수정**할 수 있습니다. 예를 들어, 파드 생성 요청이 들어오면 자동으로 사이드카 컨테이너를 주입하거나, 특정 레이블을 추가할 수 있습니다.
    -   **Validating Admission Webhooks**: 리소스 객체를 검증하고, 정책에 맞지 않으면 요청을 **거부**할 수 있습니다. 예를 들어, 모든 파드는 반드시 `owner` 레이블을 가져야 한다는 정책을 강제할 수 있습니다.
    -   **내장 컨트롤러**: `ResourceQuota` (네임스페이스의 리소스 사용량 제한), `LimitRanger` (컨테이너의 기본 리소스 요청/제한 설정) 등 다양한 내장 컨트롤러가 있습니다.

## 2. `etcd`: 클러스터의 상태 저장소

`etcd`는 클러스터의 모든 상태 정보를 저장하는 심장과도 같은 존재입니다. 모든 파드, 디플로이먼트, 서비스 등의 명세와 현재 상태가 이곳에 기록됩니다.

### Raft 합의 알고리즘

`etcd`는 여러 노드에 데이터를 복제하여 고가용성을 확보하며, 이때 데이터의 일관성을 유지하기 위해 **Raft 합의 알고리즘**을 사용합니다.

-   **리더(Leader) 선출**: `etcd` 클러스터 내에서 단 하나의 노드만이 리더가 됩니다. 모든 쓰기(Write) 요청은 리더를 통해서만 처리됩니다.
-   **로그 복제**: 리더는 데이터 변경 요청을 받으면, 이를 로그 항목으로 만든 뒤 다른 팔로워(Follower) 노드들에게 전파합니다.
-   **커밋(Commit)**: 과반수(Quorum) 이상의 노드가 로그를 성공적으로 저장했다고 응답하면, 리더는 해당 변경사항을 '커밋'하고 클라이언트에게 성공을 알립니다. 이 과반수 정책 덕분에 일부 노드가 다운되어도 클러스터는 안정적으로 동작할 수 있습니다.

### `etcdctl` 사용법

`etcd`와 직접 상호작용할 일은 드물지만, `etcdctl` CLI를 통해 상태를 확인하거나 백업/복구를 수행할 수 있습니다. (보통 마스터 노드의 파드 안에서 실행)

```bash
# etcd 멤버 목록 확인
etcdctl member list

# 전체 키(key) 목록 확인
etcdctl get / --prefix --keys-only

# 특정 파드 정보 확인
etcdctl get /registry/pods/default/my-pod-xxxx

# 스냅샷 백업
etcdctl snapshot save snapshot.db

# 스냅샷 복원
etcdctl snapshot restore snapshot.db --data-dir /var/lib/etcd-restore
```

## 3. `kube-scheduler`: 최적의 노드를 찾는 예술

`kube-scheduler`는 API 서버를 감시하다가, 아직 노드가 할당되지 않은(`nodeName` 필드가 비어있는) 파드를 발견하면 가장 적합한 노드를 찾아주는 역할을 합니다.

### 고급 스케줄링 기법

단순히 리소스만 보고 노드를 결정하는 것을 넘어, 다양한 정책을 통해 파드 배치를 정교하게 제어할 수 있습니다.

-   **Taints (오염) 와 Tolerations (용인)**
    -   `Taint`: 노드에 설정하는 '오염' 표시입니다. 특정 Taint를 용인(`Toleration`)하지 않는 파드는 해당 노드에 스케줄링될 수 없습니다.
    -   **사용 예**: GPU가 장착된 노드에 `gpu=true:NoSchedule` Taint를 설정하고, GPU를 사용해야 하는 파드에만 해당 Toleration을 추가하여 GPU 노드를 전용으로 사용하게 만들 수 있습니다.

-   **Node Affinity (노드 선호도)**
    -   파드가 특정 레이블을 가진 노드에 스케줄링되도록 유도합니다.
    -   `requiredDuringSchedulingIgnoredDuringExecution`: **반드시** 조건에 맞는 노드에만 스케줄링합니다. (Hard-affinity)
    -   `preferredDuringSchedulingIgnoredDuringExecution`: 조건에 맞는 노드를 **선호**하지만, 없을 경우 다른 노드에도 스케줄링될 수 있습니다. (Soft-affinity)
    -   **사용 예**: "SSD 디스크(`disktype=ssd`)가 장착된 노드를 선호하지만, 없어도 괜찮아."

-   **Pod Affinity / Anti-Affinity (파드 선호도/반선호도)**
    -   다른 파드와의 관계를 기반으로 스케줄링을 결정합니다.
    -   **Pod Affinity**: 특정 파드와 같은 노드/존(Zone)/리전(Region)에 배치되도록 합니다. (예: 웹 서버와 캐시 서버를 가까이 두어 네트워크 지연 감소)
    -   **Pod Anti-Affinity**: 특정 파드와 다른 노드/존/리전에 배치되도록 합니다. (예: 고가용성을 위해 데이터베이스 레플리카들을 서로 다른 노드에 분산)

## 4. `kube-controller-manager`: 상태를 유지하는 조정자

`kube-controller-manager`는 하나의 바이너리 안에 다수의 컨트롤러를 포함하고 있으며, 각 컨트롤러는 클러스터의 상태를 사용자가 의도한 상태로 유지하기 위해 끊임없이 노력합니다.

### 조정 루프 (Reconciliation Loop)

모든 컨트롤러는 **조정 루프**라는 동일한 패턴으로 동작합니다.

> 1.  **관찰(Observe)**: 자신이 담당하는 리소스의 현재 상태를 API 서버를 통해 관찰합니다.
> 2.  **비교(Compare)**: 현재 상태와 리소스의 명세(`.spec`)에 정의된 '의도한 상태'를 비교합니다.
> 3.  **조치(Act)**: 두 상태 간에 차이가 있다면, 이를 해소하기 위한 작업을 수행합니다. (예: 파드 생성, 서비스 엔드포인트 업데이트 등)

### 주요 컨트롤러 예시

-   **Deployment Controller**: `Deployment` 객체를 관리합니다. 사용자가 레플리카 수를 3에서 5로 변경하면, 이 컨트롤러가 `ReplicaSet`을 새로 만들고, 이전 `ReplicaSet`의 파드를 점진적으로 줄여 롤링 업데이트를 수행합니다.
-   **Namespace Controller**: `Namespace`가 삭제되기를 기다릴 때, 해당 네임스페이스 내의 모든 리소스가 삭제될 때까지 기다렸다가 최종적으로 네임스페이스를 삭제합니다.
-   **EndpointSlice Controller**: `Service`의 셀렉터에 매칭되는 파드들의 IP와 포트 목록을 `EndpointSlice` 객체에 최신 상태로 유지하여 서비스 디스커버리가 가능하게 합니다.

## 결론

컨트롤 플레인은 쿠버네티스의 안정성, 확장성, 보안을 책임지는 정교한 시스템입니다. `kube-apiserver`는 중앙 관제탑, `etcd`는 블랙박스, `kube-scheduler`는 배차 담당관, `kube-controller-manager`는 자동 복구 시스템 역할을 하며 유기적으로 동작합니다. 이들의 내부 동작을 이해하면 쿠버네티스에서 발생하는 문제를 진단하고, 더 효율적으로 클러스터를 운영하는 데 큰 도움이 될 것입니다.

다음 2부에서는 이 컨트롤 플레인의 지시를 받아 실제 작업을 수행하는 **워커 노드**의 컴포넌트들을 심층 분석하겠습니다.
