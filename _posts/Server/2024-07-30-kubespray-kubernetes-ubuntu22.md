---
title: Kubespray를 이용한 Kubernetes 클러스터 설치 (Ubuntu 22.04)
excerpt: Ubuntu 22.04 환경에서 Kubespray를 사용하여 고가용성 Kubernetes 클러스터를 구축하는 방법
categories:
- server
- general
tags:
- - Kubernetes
  - Kubespray
  - Ubuntu
  - DevOps
  - Container
  - Server
toc: true
toc_sticky: true
date: 2024-07-30
last_modified_at: 2024-07-30
---

## Kubespray란?

Kubespray는 Ansible을 기반으로 한 Kubernetes 클러스터 배포 도구입니다. 프로덕션 환경에서 사용할 수 있는 고가용성 Kubernetes 클러스터를 자동화된 방식으로 설치할 수 있게 해줍니다.

### Kubespray의 주요 특징

- **Ansible 기반**: Ansible 플레이북을 사용하여 멱등성(idempotent) 보장
- **다양한 OS 지원**: Ubuntu, CentOS, RHEL, Debian 등 다양한 운영체제 지원
- **고가용성**: 여러 마스터 노드를 통한 HA 구성 지원
- **CNI 플러그인**: Calico, Flannel, Weave Net 등 다양한 네트워크 플러그인 지원
- **커스터마이징**: 다양한 설정 옵션을 통한 클러스터 커스터마이징

## 사전 요구사항

### 하드웨어 요구사항

#### 마스터 노드
- CPU: 최소 2 코어 (권장 4 코어)
- RAM: 최소 4GB (권장 8GB)
- 디스크: 최소 20GB

#### 워커 노드
- CPU: 최소 1 코어 (권장 2 코어)
- RAM: 최소 2GB (권장 4GB)
- 디스크: 최소 20GB

### 네트워크 요구사항

- 모든 노드 간 SSH 접근 가능
- 인터넷 연결 (패키지 다운로드용)
- 방화벽 포트 개방 (Kubernetes API, etcd, CNI 등)

## 환경 준비

### 1. 서버 준비

이 예제에서는 다음과 같은 구성을 사용합니다:

```bash
# 마스터 노드 (Control Plane)
master01: 192.168.1.10
master02: 192.168.1.11
master03: 192.168.1.12

# 워커 노드
worker01: 192.168.1.20
worker02: 192.168.1.21
```

### 2. Ubuntu 22.04 기본 설정

모든 노드에서 다음 작업을 수행합니다:

```bash
# 시스템 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
sudo apt install -y curl wget vim git

# 시간 동기화 설정
sudo timedatectl set-timezone Asia/Seoul
sudo systemctl enable systemd-timesyncd
sudo systemctl start systemd-timesyncd

# 스왑 비활성화
sudo swapoff -a
sudo sed -i '/swap/d' /etc/fstab

# 방화벽 비활성화 (또는 필요한 포트만 개방)
sudo ufw disable
```

### 3. SSH 키 기반 인증 설정

Ansible 노드(배포를 실행할 서버)에서:

```bash
# SSH 키 생성
ssh-keygen -t rsa -b 4096 -C "your-email@example.com"

# 모든 노드에 공개키 복사
for host in 192.168.1.10 192.168.1.11 192.168.1.12 192.168.1.20 192.168.1.21; do
    ssh-copy-id user@$host
done
```

## Kubespray 설치 및 설정

### 1. Kubespray 다운로드

```bash
# Git 클론
git clone https://github.com/kubernetes-sigs/kubespray.git
cd kubespray

# 안정적인 버전으로 체크아웃 (선택사항)
git checkout release-2.24

# Python 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 인벤토리 설정

```bash
# 샘플 인벤토리 복사
cp -r inventory/sample inventory/mycluster

# 인벤토리 파일 편집
vim inventory/mycluster/inventory.ini
```

`inventory/mycluster/inventory.ini` 파일 내용:

```ini
[all]
master01 ansible_host=192.168.1.10 ip=192.168.1.10
master02 ansible_host=192.168.1.11 ip=192.168.1.11
master03 ansible_host=192.168.1.12 ip=192.168.1.12
worker01 ansible_host=192.168.1.20 ip=192.168.1.20
worker02 ansible_host=192.168.1.21 ip=192.168.1.21

[kube_control_plane]
master01
master02
master03

[etcd]
master01
master02
master03

[kube_node]
worker01
worker02

[calico_rr]

[k8s_cluster:children]
kube_control_plane
kube_node
calico_rr
```

### 3. 클러스터 설정 커스터마이징

```bash
# 그룹 변수 파일 편집
vim inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml
```

주요 설정 옵션:

```yaml
# Kubernetes 버전
kube_version: v1.28.2

# 클러스터 이름
cluster_name: mycluster.local

# 네트워크 플러그인
kube_network_plugin: calico

# Service 서브넷
kube_service_addresses: 10.233.0.0/18

# Pod 서브넷
kube_pods_subnet: 10.233.64.0/18

# DNS 도메인
cluster_name: cluster.local

# 로드밸런서 설정 (HA 구성 시)
loadbalancer_apiserver_localhost: true
```

### 4. 추가 설정 (선택사항)

#### Ingress Controller 활성화
```bash
vim inventory/mycluster/group_vars/k8s_cluster/addons.yml
```

```yaml
# Nginx Ingress Controller
ingress_nginx_enabled: true
ingress_nginx_host_network: true

# MetalLB (베어메탈 환경에서 LoadBalancer 서비스 지원)
metallb_enabled: true
metallb_speaker_enabled: true
metallb_ip_range:
  - "192.168.1.100-192.168.1.200"
```

## Kubernetes 클러스터 배포

### 1. 연결 테스트

```bash
# 모든 노드에 연결 가능한지 확인
ansible all -i inventory/mycluster/inventory.ini -m ping
```

### 2. 클러스터 배포 실행

```bash
# 클러스터 배포 (약 20-30분 소요)
ansible-playbook -i inventory/mycluster/inventory.ini --become --become-user=root cluster.yml
```

### 3. 배포 진행 상황 모니터링

배포 과정에서 다음과 같은 단계들이 실행됩니다:

1. **사전 검사**: 시스템 요구사항 확인
2. **Docker/containerd 설치**: 컨테이너 런타임 설치
3. **etcd 클러스터 구성**: 분산 키-값 저장소 설정
4. **Kubernetes 컴포넌트 설치**: kubelet, kubeadm, kubectl 설치
5. **Control Plane 초기화**: 마스터 노드 설정
6. **워커 노드 조인**: 워커 노드를 클러스터에 추가
7. **네트워크 플러그인 설치**: CNI 플러그인 구성
8. **DNS 설정**: CoreDNS 구성

## 클러스터 검증

### 1. kubectl 설정

마스터 노드에서:

```bash
# kubectl 설정 파일 복사
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

### 2. 클러스터 상태 확인

```bash
# 노드 상태 확인
kubectl get nodes -o wide

# 시스템 파드 확인
kubectl get pods -n kube-system

# 클러스터 정보 확인
kubectl cluster-info

# 컴포넌트 상태 확인
kubectl get componentstatuses
```

예상 출력:

```bash
$ kubectl get nodes
NAME       STATUS   ROLES           AGE   VERSION
master01   Ready    control-plane   5m    v1.28.2
master02   Ready    control-plane   4m    v1.28.2
master03   Ready    control-plane   4m    v1.28.2
worker01   Ready    <none>          3m    v1.28.2
worker02   Ready    <none>          3m    v1.28.2
```

### 3. 테스트 애플리케이션 배포

```bash
# 테스트 Pod 생성
kubectl create deployment nginx --image=nginx

# 서비스 노출
kubectl expose deployment nginx --port=80 --type=NodePort

# 상태 확인
kubectl get pods,svc
```

## 문제해결

### 일반적인 문제들

#### 1. SSH 연결 실패
```bash
# SSH 키 권한 확인
chmod 600 ~/.ssh/id_rsa
chmod 700 ~/.ssh

# SSH 에이전트 실행
eval $(ssh-agent)
ssh-add ~/.ssh/id_rsa
```

#### 2. 방화벽 문제
Ubuntu 22.04에서 필요한 포트들:

```bash
# 마스터 노드
sudo ufw allow 6443/tcp    # Kubernetes API
sudo ufw allow 2379:2380/tcp  # etcd
sudo ufw allow 10250/tcp   # kubelet
sudo ufw allow 10251/tcp   # kube-scheduler
sudo ufw allow 10252/tcp   # kube-controller-manager

# 워커 노드
sudo ufw allow 10250/tcp   # kubelet
sudo ufw allow 30000:32767/tcp  # NodePort services
```

#### 3. 메모리 부족
```bash
# 스왑 확인
sudo swapon --show

# 스왑 영구 비활성화 확인
cat /etc/fstab | grep swap
```

#### 4. 배포 실패 시 재시도
```bash
# 클러스터 리셋
ansible-playbook -i inventory/mycluster/inventory.ini --become --become-user=root reset.yml

# 재배포
ansible-playbook -i inventory/mycluster/inventory.ini --become --become-user=root cluster.yml
```

## 클러스터 관리

### 1. 노드 추가

새로운 워커 노드를 추가할 때:

```bash
# inventory.ini에 새 노드 추가
vim inventory/mycluster/inventory.ini

# 스케일 아웃 실행
ansible-playbook -i inventory/mycluster/inventory.ini --become --become-user=root scale.yml
```

### 2. 클러스터 업그레이드

```bash
# 새 버전으로 설정 변경
vim inventory/mycluster/group_vars/k8s_cluster/k8s-cluster.yml

# 업그레이드 실행
ansible-playbook -i inventory/mycluster/inventory.ini --become --become-user=root upgrade-cluster.yml
```

### 3. 백업

```bash
# etcd 백업
sudo ETCDCTL_API=3 etcdctl snapshot save backup.db \
  --endpoints=https://127.0.0.1:2379 \
  --cacert=/etc/ssl/etcd/ssl/ca.pem \
  --cert=/etc/ssl/etcd/ssl/admin-master01.pem \
  --key=/etc/ssl/etcd/ssl/admin-master01-key.pem
```

## 마무리

Kubespray를 사용하면 Ubuntu 22.04 환경에서 프로덕션 급 Kubernetes 클러스터를 비교적 쉽게 구축할 수 있습니다. 

주요 장점:
- **자동화**: 반복 가능한 배포 프로세스
- **확장성**: 노드 추가/제거 용이
- **안정성**: 검증된 구성과 베스트 프랙티스 적용
- **커스터마이징**: 다양한 설정 옵션 제공

정기적인 백업과 모니터링을 통해 클러스터를 안정적으로 운영하시기 바랍니다.