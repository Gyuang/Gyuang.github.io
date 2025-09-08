---
title: "KEGG PATHWAY 분류 체계: 생명현상의 분자적 지도"
excerpt: "KEGG에서 제공하는 500여 개 생물학적 경로들의 체계적 분류와 상세 분석"

categories:
  - Medical AI
tags:
  - [KEGG, Pathway, Metabolism, Signal Transduction, Systems Biology, Medical AI]

toc: true
toc_sticky: true

date: 2025-09-08
last_modified_at: 2025-09-08

---

## KEGG PATHWAY 분류 체계 개요

KEGG PATHWAY는 현재 500개 이상의 참조 경로(reference pathway)를 포함하고 있으며, 이들은 생명체의 모든 주요 생물학적 과정을 망라합니다. 각 경로는 고유한 맵 번호(map number)를 가지며, 기능적 범주에 따라 체계적으로 분류되어 있습니다.

### 전체 분류 구조
KEGG PATHWAY는 다음과 같은 6개 주요 범주로 구분됩니다:

1. **Metabolism (대사)** - map00001~map01999
2. **Genetic Information Processing (유전정보처리)** - map03000~map03999  
3. **Environmental Information Processing (환경정보처리)** - map04000~map04999
4. **Cellular Processes (세포과정)** - map05000~map05999
5. **Organismal Systems (개체 시스템)** - map06000~map06999
6. **Human Diseases (인간 질병)** - map05000~map05999

## 1. Metabolism (대사) - 기본 생명 활동의 화학적 기반

대사 경로는 KEGG PATHWAY의 가장 큰 범주로, 생명체의 에너지 생산과 생체분자 합성에 관련된 모든 화학 반응을 포함합니다.

### 1.1 Carbohydrate Metabolism (탄수화물 대사)

#### 해당과정/당신생합성 (Glycolysis/Gluconeogenesis) - map00010
```
포도당 → 피루브산 (해당과정)
피루브산 → 포도당 (당신생합성)

주요 효소:
- Hexokinase (EC 2.7.1.1): 포도당 → 포도당-6-인산
- Phosphofructokinase (EC 2.7.1.11): 과당-6-인산 → 과당-1,6-이인산
- Pyruvate kinase (EC 2.7.1.40): 포스포엔올피루브산 → 피루브산
- Glucose-6-phosphatase (EC 3.1.3.9): 포도당-6-인산 → 포도당

조절 메커니즘:
- Allosteric regulation: PFK의 ATP 저해, AMP 활성화
- Covalent modification: 인산화/탈인산화
- Transcriptional control: SREBP-1c, ChREBP

질병 연관성:
- 당뇨병: 인슐린 저항성으로 인한 당신생합성 증가
- 암: Warburg effect로 인한 해당과정 활성화
```

#### 시트르산 회로 (TCA cycle) - map00020
```
아세틸-CoA → CO2 + NADH + FADH2 + GTP

8단계 순환 반응:
1. 시트르산 형성: 아세틸-CoA + 옥살아세트산 → 시트르산
2. 이소시트르산 형성: 시트르산 → 이소시트르산  
3. α-케토글루타르산 형성: 이소시트르산 → α-케토글루타르산 + NADH
4. 숙시닐-CoA 형성: α-케토글루타르산 → 숙시닐-CoA + NADH
5. 숙신산 형성: 숙시닐-CoA → 숙신산 + GTP
6. 푸마르산 형성: 숙신산 → 푸마르산 + FADH2
7. 말산 형성: 푸마르산 → 말산
8. 옥살아세트산 재생: 말산 → 옥살아세트산 + NADH

조절 효소:
- Citrate synthase: 아세틸-CoA 가용성에 의존
- Isocitrate dehydrogenase: Ca2+에 의해 활성화
- α-Ketoglutarate dehydrogenase: 숙시닐-CoA에 의해 저해

생합성 전구체 제공:
- 아미노산 합성: α-케토글루타르산, 옥살아세트산
- 지방산 합성: 아세틸-CoA
- 콜레스테롤 합성: 아세틸-CoA
```

#### 오탄당 인산 경로 (Pentose phosphate pathway) - map00030
```
포도당-6-인산 → 리보스-5-인산 + 2NADPH + CO2

산화 단계 (비가역):
1. 포도당-6-인산 → 6-포스포글루콘산 + NADPH
2. 6-포스포글루콘산 → 리불로스-5-인산 + NADPH + CO2

비산화 단계 (가역):
3. 리불로스-5-인산 → 리보스-5-인산/자일룰로스-5-인산
4. 당 재배열 반응으로 C3, C4, C6, C7 당들 상호변환

생물학적 중요성:
- NADPH 생산: 지방산 합성, 항산화 방어
- 리보스-5-인산 생산: 핵산 합성
- 적혈구: 글루타티온 환원을 통한 산화 스트레스 방어

질병:
- G6PD 결핍증: 용혈성 빈혈
- 암세포: NADPH 요구량 증가로 활성 증가
```

### 1.2 Energy Metabolism (에너지 대사)

#### 산화적 인산화 (Oxidative phosphorylation) - map00190
```
NADH/FADH2 → ATP + H2O

전자전달사슬:
- Complex I (NADH dehydrogenase): NADH → CoQ
- Complex II (Succinate dehydrogenase): FADH2 → CoQ  
- Complex III (Cytochrome bc1): CoQ → Cytochrome c
- Complex IV (Cytochrome oxidase): Cytochrome c → O2

ATP 합성:
- Complex V (ATP synthase): ADP + Pi → ATP

화학삼투 이론:
- 프로톤 구배: 매트릭스 vs 막간공간
- 프로톤 동력: ΔμH+ = Δψ + 2.3RT/F × ΔpH
- P/O ratio: NADH = 2.5, FADH2 = 1.5

미토콘드리아 질환:
- LHON (Leber hereditary optic neuropathy)
- MELAS (Mitochondrial encephalomyopathy)
- Complex I 결핍증
```

#### 광합성 (Photosynthesis) - map00195
```
6CO2 + 6H2O + 빛에너지 → C6H12O6 + 6O2

명반응 (틸라코이드):
- Photosystem II: H2O → O2 + 4H+ + 4e-
- 전자전달사슬: PSII → PQ → Cyt bf → PC → PSI
- Photosystem I: 전자 → NADP+ → NADPH
- ATP synthase: ADP + Pi → ATP

암반응 (스트로마):
- Calvin cycle: CO2 고정
- RuBisCO: RuBP + CO2 → 2 × 3PG
- 환원: 3PG → G3P (NADPH 소모)
- 재생: G3P → RuBP (ATP 소모)

C4 광합성:
- 공간적 분리: 엽육세포 vs 유관속초세포
- CO2 농축: 말산/아스파트산 수송
- 효율성: 고온/건조 환경 적응

CAM 광합성:
- 시간적 분리: 밤 vs 낮
- 말산 저장: 액포에 야간 저장
- 수분 보존: 사막 식물 적응
```

### 1.3 Lipid Metabolism (지질 대사)

#### 지방산 대사 (Fatty acid metabolism) - map01212
```
지방산 분해 (β-oxidation):
팔미트산 (C16:0) → 8 아세틸-CoA + 7 FADH2 + 7 NADH

단계:
1. 활성화: 지방산 + CoA → 아실-CoA (ATP 소모)
2. 수송: 카르니틴 셔틀로 미토콘드리아 매트릭스 이동
3. β-oxidation: 4단계 반복
   - 산화: 아실-CoA → 에노일-CoA (FADH2 생산)
   - 수화: 에노일-CoA → 3-히드록시아실-CoA  
   - 산화: 3-히드록시아실-CoA → 3-케토아실-CoA (NADH 생산)
   - 절단: 3-케토아실-CoA → 아세틸-CoA + 아실-CoA(n-2)

지방산 합성:
아세틸-CoA → 팔미트산

단계:
1. 아세틸-CoA carboxylase: 아세틸-CoA → 말로닐-CoA
2. Fatty acid synthase: 7회 반복으로 C16 지방산 합성
   - 응축: 아세틸-ACP + 말로닐-ACP → 아세토아세틸-ACP
   - 환원: 아세토아세틸-ACP → D-3-히드록시부티릴-ACP (NADPH)
   - 탈수: D-3-히드록시부티릴-ACP → 크로토닐-ACP
   - 환원: 크로토닐-ACP → 부티릴-ACP (NADPH)

조절:
- ACC: 시트르산 활성화, 팔미토일-CoA 저해
- 인슐린: ACC 탈인산화 (활성화)
- 글루카곤: ACC 인산화 (비활성화)
```

#### 콜레스테롤 대사 (Cholesterol metabolism) - map04979
```
콜레스테롤 합성 (메발론산 경로):
3 아세틸-CoA → 콜레스테롤

주요 단계:
1. HMG-CoA 형성: 3 아세틸-CoA → HMG-CoA
2. 메발론산 형성: HMG-CoA → 메발론산 (HMG-CoA reductase)
3. 이소펜테닐 피로인산 형성: 메발론산 → IPP
4. 스쿠알렌 형성: 6 IPP → 스쿠알렌
5. 라노스테롤 형성: 스쿠알렌 → 라노스테롤
6. 콜레스테롤 형성: 라노스테롤 → 콜레스테롤 (19단계)

조절 메커니즘:
- SREBP-2: 콜레스테롤 저하 시 HMG-CoA reductase 전사 증가
- 피드백 저해: 콜레스테롤에 의한 HMG-CoA reductase 저해
- 상위 조절: 인슐린 활성화, 글루카곤 저해

담즙산 합성:
콜레스테롤 → 담즙산

일차 담즙산:
- 콜산 (cholic acid)
- 케노데옥시콜산 (chenodeoxycholic acid)

이차 담즙산 (장내 세균):
- 데옥시콜산
- 리토콜산

질병 연관성:
- 가족성 고콜레스테롤혈증: LDL 수용체 결함
- 죽상동맥경화증: 콜레스테롤 축적
- 담석증: 콜레스테롤 침전
```

### 1.4 Amino Acid Metabolism (아미노산 대사)

#### 필수 아미노산 생합성 - map01230
```
인간은 합성할 수 없어 식이로 섭취해야 하는 9개 아미노산:
히스티딘, 이소류신, 류신, 라이신, 메티오닌, 페닐알라닌, 
트레오닌, 트립토판, 발린

식물/미생물에서의 합성 경로:

류신 생합성:
피루브산 → α-케토이소발레르산 → 류신
- 브랜치드 체인 아미노산 (BCAA) 중 하나
- 단백질 합성, mTOR 활성화

라이신 생합성:
아스파르트산 → 디아미노피멜산 → 라이신
- 히스톤 메틸화의 주요 표적
- 콜라겐 교차결합 형성

메티오닌 생합성:
아스파르트산 → 호모시스테인 → 메티오닌
- S-아데노실메티오닌 (SAM) 전구체
- 메틸화 반응의 메틸 공여체

질병 연관성:
- 메이플시럽요증: BCAA 분해 효소 결함
- 호모시스틴뇨증: 메티오닌 대사 장애
- 페닐케톤뇨증: 페닐알라닌 대사 결함
```

#### 요소 회로 (Urea cycle) - map00220
```
암모니아 해독과 질소 배설 경로:
NH3 + CO2 → 요소 + H2O

5단계 반응:
1. 카르바모일 인산 형성: NH3 + CO2 → 카르바모일 인산 (미토콘드리아)
2. 시트룰린 형성: 카르바모일 인산 + 오르니틴 → 시트룰린
3. 아르기니노숙신산 형성: 시트룰린 + 아스파르트산 → 아르기니노숙신산 (세포질)
4. 아르기닌 형성: 아르기니노숙신산 → 아르기닌 + 푸마르산
5. 요소 형성: 아르기닌 → 요소 + 오르니틴

조절:
- N-아세틸글루타메이트: CPS I 활성화
- 단백질 섭취량에 따른 효소 유도
- 간 질환 시 기능 저하

요소 회로 결함증:
- OTC 결핍증: 고암모니아혈증
- 시트룰린혈증: 아르기니노숙신산 합성효소 결함
- 아르기닌혈증: 아르기나제 결함

치료:
- 단백질 제한식
- 질소 청소제 (sodium benzoate, sodium phenylacetate)
- 아르기닌/시트룰린 보충
```

## 2. Genetic Information Processing (유전정보처리)

유전자에서 단백질까지의 정보 전달 과정과 관련된 경로들입니다.

### 2.1 Transcription (전사)

#### RNA 중합효소 (RNA polymerase) - map03020
```
원핵생물 RNA 중합효소:
- 코어 효소: α2ββ'ω
- 홀로효소: 코어 + σ 인자
- σ70: 하우스키핑 유전자
- σ32: 열충격 반응
- σ54: 질소 대사

진핵생물 RNA 중합효소:
- RNA Pol I: rRNA 합성 (45S pre-rRNA)
- RNA Pol II: mRNA, miRNA, lncRNA 합성
- RNA Pol III: tRNA, 5S rRNA, U6 snRNA 합성

전사 과정:
1. 개시 (Initiation):
   - 프로모터 인식: TFIID → TFIIA/TFIIB → TFIIE/TFIIF → TFIIH
   - 전전사 복합체 형성
   - DNA 해리 및 전사 개시

2. 연장 (Elongation):
   - CTD 인산화: Ser2, Ser5, Ser7
   - 전사 연장 인자: TFIIS, P-TEFb
   - 뉴클레오솜 리모델링

3. 종료 (Termination):
   - Intrinsic termination: 헤어핀 구조
   - Rho-dependent termination
   - Polyadenylation signal (AAUAAA)

조절 메커니즘:
- 프로모터: 기본 전사
- 인핸서: 전사 활성화
- 사일런서: 전사 억제
- 인슐레이터: 도메인 경계
```

#### 전사 조절 인자 (Transcription factors) - map03000
```
전사 조절 인자 분류:

DNA 결합 도메인별:
1. Helix-turn-helix: 호메오도메인, HTH
2. Zinc finger: C2H2, C4, C6
3. Leucine zipper: bZIP (AP1, CREB)
4. Helix-loop-helix: bHLH (MyoD, E2A)
5. Beta sheet: β-barrel

기능별:
1. 활성화 인자 (Activators):
   - p53: DNA 손상 반응
   - NF-κB: 염증 반응
   - SREBP: 지질 대사
   - HIF-1: 저산소 반응

2. 억제 인자 (Repressors):
   - Rb: 세포주기 억제
   - p16: CDK4/6 억제
   - Mad: Myc 길항

3. 구조 단백질:
   - TFIID: 기본 전사 기계
   - Mediator: 전사 조절 복합체
   - Cohesion: 염색체 구조

신호 의존적 활성화:
- cAMP → PKA → CREB 인산화 → CRE 결합
- 성장인자 → MAPK → Elk1 인산화 → SRE 결합
- 스테로이드 → 핵수용체 → HRE 결합

질병 연관성:
- p53 돌연변이: 암의 50%
- BRCA1/2: 유방암/난소암
- APC: 대장암
- VHL: 신장암
```

### 2.2 Translation (번역)

#### 리보솜 (Ribosome) - map03010
```
리보솜 구조:
- 원핵생물: 70S (30S + 50S)
- 진핵생물: 80S (40S + 60S)

30S 서브유닛 (16S rRNA + 21개 단백질):
- 16S rRNA: 디코딩 중심
- S1: mRNA 결합
- S3, S4, S5: 정확성 검증

50S 서브유닛 (23S, 5S rRNA + 31개 단백질):
- 23S rRNA: 펩티딜 전이효소 활성
- L1, L7/L12: 방출 인자 결합
- L22: 펩티드 터널

번역 과정:
1. 개시 (Initiation):
   - mRNA 리보솜 결합부위 (RBS) 인식
   - 개시 코돈 (AUG) 검색
   - Met-tRNA 결합

2. 연장 (Elongation):
   - EF-Tu: 아미노아실-tRNA 전달
   - 펩티드 결합 형성
   - EF-G: 리보솜 전이

3. 종료 (Termination):
   - 종료 코돈 (UAG, UAA, UGA) 인식
   - eRF1: 펩티딜 가수분해
   - eRF3: GTPase 활성

품질 관리:
- Nonsense-mediated decay: 조기 종료 코돈
- No-go decay: 리보솜 정지
- Nonstop decay: 종료 코돈 부재
```

#### 아미노아실-tRNA 생합성 - map00970
```
20개 아미노아실-tRNA 합성효소:

Class I (10개):
- 단량체 구조
- HIGH/KMSKS 모티프
- tRNA 3' 말단에 아미노산 부착

Class II (10개):  
- 동이량체 구조
- 모티프 1, 2, 3
- tRNA 2' 말단에 아미노산 부착

아미노아실화 반응:
1단계: 아미노산 + ATP → 아미노아실-AMP + PPi
2단계: 아미노아실-AMP + tRNA → 아미노아실-tRNA + AMP

정확성 메커니즘:
1. Initial selection: 아미노산 특이성
2. Proofreading: 잘못된 아미노산 제거
3. Post-transfer editing: 아미노아실-tRNA 수정

오류율: 10^-4 ~ 10^-5

특수한 경우:
- Selenocysteine: UGA 코돈, SECIS 요소
- Pyrrolysine: UAG 코돈, pylT tRNA
- 미토콘드리아: 별도의 합성효소 세트

질병:
- Charcot-Marie-Tooth: 말초신경병증
- Microcephaly: 뇌 발달 장애
```

### 2.3 DNA Replication and Repair (DNA 복제와 수선)

#### DNA 복제 (DNA replication) - map03030
```
반보존적 복제:
5' → 3' 방향 합성

주요 효소:
1. Helicase: DNA 해리 (DnaB, MCM2-7)
2. Primase: RNA 프라이머 합성
3. DNA polymerase:
   - Pol α: 프라이머 합성
   - Pol δ: 후행 가닥 합성  
   - Pol ε: 선행 가닥 합성
4. DNA ligase: 옥수아키 절편 연결

복제 개시:
- Origin recognition complex (ORC)
- Pre-replication complex 형성
- S phase 진입: CDK2 활성화

선행/후행 가닥:
- Leading strand: 연속 합성
- Lagging strand: 불연속 합성 (Okazaki fragments)

복제포크 진행:
- 프로세시비티: 한 번에 합성하는 염기 수
- Sliding clamp (PCNA): 프로세시비티 증가
- Replication fork barrier: rDNA, 텔로미어

텔로미어 문제:
- 말단 복제 문제
- 텔로머라제: TERT + TERC
- 간질세포: 텔로머라제 활성
- 체세포: 텔로미어 단축 → 노화
```

#### DNA 수선 (DNA repair) - map03410
```
DNA 손상 유형과 수선 기전:

1. Base excision repair (BER):
   - 손상: 탈퓨린화, 탈아민화, 산화
   - 효소: DNA glycosylase → AP endonuclease → Pol β → ligase

2. Nucleotide excision repair (NER):
   - 손상: UV 손상, 화학적 부가체
   - 전역 NER: 전체 게놈
   - 전사 결합 NER: 활성 유전자

3. Mismatch repair (MMR):
   - 손상: 염기 부정합, 루프
   - 효소: MSH2/MSH6 → MLH1/PMS2 → 절제 및 재합성

4. Homologous recombination (HR):
   - 손상: 이중 가닥 절단
   - 효소: RAD51, BRCA1, BRCA2
   - 정확한 수선

5. Non-homologous end joining (NHEJ):
   - 손상: 이중 가닥 절단
   - 효소: Ku70/80, DNA-PKcs, ligase IV
   - 오류 발생 가능

DNA 손상 체크포인트:
- ATM/ATR 키나제 활성화
- p53 안정화 및 활성화
- p21 유도 → 세포주기 정지
- PARP: DNA 손상 감지 및 수선 촉진

질병:
- 색소건피증: NER 결함
- 유전성 비용종 대장암: MMR 결함
- 유방암/난소암: BRCA1/2 결함
- 아토피아혈관확장증: ATM 결함
```

## 3. Environmental Information Processing (환경정보처리)

세포가 외부 환경과 상호작용하고 신호를 처리하는 과정입니다.

### 3.1 Signal Transduction (신호전달)

#### MAPK 신호전달 경로 - map04010
```
3단계 키나제 캐스케이드:
MAPKKK → MAPKK → MAPK

주요 MAPK 경로:

1. ERK 경로 (Ras/Raf/MEK/ERK):
   자극: 성장인자, 미토겐
   신호: RTK → Ras → Raf → MEK1/2 → ERK1/2
   표적: c-Fos, c-Jun, Elk1
   기능: 세포 증식, 분화

2. JNK 경로:
   자극: 스트레스, 사이토카인
   신호: MEKK → MKK4/7 → JNK1/2/3
   표적: c-Jun, ATF2
   기능: 스트레스 반응, 세포사멸

3. p38 경로:
   자극: 스트레스, 염증
   신호: ASK1/TAK1 → MKK3/6 → p38
   표적: ATF2, CREB, p53
   기능: 염증, 분화, 세포사멸

조절 메커니즘:
- DUSP: 이중 특이성 인산화효소
- Scaffolding proteins: KSR, JIP
- Compartmentalization: 세포내 국재화

질병 연관성:
- 암: ERK 과활성화
- 염증성 질환: p38/JNK 활성화
- 신경퇴행성 질환: JNK 활성화
```

#### PI3K-Akt 신호전달 경로 - map04151
```
신호 흐름:
RTK/GPCR → PI3K → PIP3 → PDK1/mTORC2 → Akt → 하위 표적

PI3K 분류:
- Class I: p110α/β/γ/δ + p85/p101 조절 서브유닛
- Class II: PI3K-C2α/β/γ
- Class III: Vps34

Akt 동이소형:
- Akt1: 세포 생존, 대사
- Akt2: 포도당 대사
- Akt3: 뇌 발달

주요 Akt 표적:
1. 세포 생존:
   - BAD 인산화 → 14-3-3 결합 → 생존
   - FoxO 인산화 → 핵 배출 → 생존

2. 단백질 합성:
   - mTOR 활성화 → S6K1, 4E-BP1
   - GSK3β 인산화 → 단백질 합성 증가

3. 포도당 대사:
   - AS160 인산화 → GLUT4 전위
   - GSK3β 인산화 → 글리코겐 합성

4. 세포주기:
   - p21 인산화 → 분해
   - p27 인산화 → 핵 배출

음성 조절:
- PTEN: PIP3 → PIP2
- SHIP: PIP3 → PIP2
- PP2A: Akt 탈인산화

질병:
- 당뇨병: 인슐린 저항성
- 암: PI3K 활성화, PTEN 소실
- 비만: mTOR 과활성화
```

#### Wnt 신호전달 경로 - map04310
```
정준 Wnt 경로 (β-catenin 의존):

1. Wnt 부재 시:
   - β-catenin 분해 복합체: APC, Axin, GSK3β, CK1
   - β-catenin 인산화 → 유비퀴틴화 → 프로테아솜 분해

2. Wnt 존재 시:
   - Wnt → Frizzled/LRP5/6 → Dishevelled 활성화
   - 분해 복합체 해체 → β-catenin 안정화
   - β-catenin 핵 전위 → TCF/LEF → 표적 유전자 활성화

표적 유전자:
- c-Myc: 세포 증식
- Cyclin D1: 세포주기 진행
- CD44: 세포 접착
- Survivin: 세포 생존

비정준 경로:
1. PCP (Planar cell polarity):
   - Wnt5a/11 → Frizzled → Dishevelled → JNK

2. Ca2+ 경로:
   - Wnt5a → Frizzled → IP3/DAG → Ca2+ 증가

발생에서의 역할:
- 축 형성: Wnt3, Wnt8
- 신경 발달: Wnt1, Wnt3a
- 간질세포 유지: Wnt3a

질병:
- 대장암: APC 돌연변이 (85%)
- 간암: β-catenin 돌연변이
- 골다공증: LRP5 돌연변이
- 정신분열병: Disc1-Wnt 상호작용
```

### 3.2 Membrane Transport (막 수송)

#### ABC 운반체 (ABC transporters) - map02010
```
ATP 결합 카세트 운반체:
구조: (TMD-NBD)2 또는 TMD-NBD-TMD-NBD

주요 ABC 운반체:

1. ABCB1 (P-glycoprotein/MDR1):
   - 기능: 다약물 배출
   - 기질: 항암제, 항생제
   - 위치: 혈뇌장벽, 장상피
   - 질병: 약물 저항성

2. ABCC7 (CFTR):
   - 기능: 클로라이드 채널
   - 조절: cAMP/PKA 인산화
   - 위치: 상피세포
   - 질병: 낭포성 섬유증

3. ABCA1:
   - 기능: 콜레스테롤/인지질 배출
   - 표적: ApoA1
   - 조절: LXR 활성화
   - 질병: Tangier disease

4. ABCG2 (BCRP):
   - 기능: 다약물 저항성
   - 기질: 미톡산트론, 이리노테칸
   - 발현: 줄기세포, 태반

조절 메커니즘:
- 전사: 핵수용체 (PXR, CAR, FXR)
- 후전사: miRNA, lncRNA
- 번역후: 인산화, 유비퀴틴화

임상적 의의:
- 약물 동역학: 흡수, 분포, 배설
- 약물 상호작용: 억제제/유도제
- 개인차: 유전자 다형성
- 표적 치료: ABC 운반체 조절제
```

## 4. Cellular Processes (세포과정)

세포의 기본적인 생명 활동과 관련된 과정들입니다.

### 4.1 Cell Growth and Death (세포 성장과 사멸)

#### 세포주기 (Cell cycle) - map04110
```
G1 → S → G2 → M → G1

주요 조절 인자:

G1/S 체크포인트:
- Cyclin D + CDK4/6: Rb 인산화
- Rb 불활성화 → E2F 방출 → S상 유전자 발현
- Cyclin E + CDK2: S상 진입

S상:
- DNA 복제
- 히스톤 합성
- Cyclin A + CDK2

G2/M 체크포인트:
- Cyclin B + CDK1: 핵막 붕괴, 염색체 응축
- Aurora kinase: 중심체 성숙
- PLK1: 중심체 분리

M상 체크포인트 (Spindle checkpoint):
- APC/C 활성화: Cyclin B 분해
- Separase 활성화: 자매염색분체 분리
- Cytokinesis: 세포질 분열

CDK 저해제:
- CIP/KIP 가족: p21, p27, p57
- INK4 가족: p15, p16, p18, p19

종양억제인자:
- p53: DNA 손상 시 p21 유도
- Rb: E2F 억제
- p16: CDK4/6 억제

암유전자:
- c-Myc: Cyclin D1 유도
- Cyclin D1: 과발현 → 세포주기 촉진
- CDK4: 증폭 → Rb 과인산화
```

#### 세포자살 (Apoptosis) - map04210
```
내재적 경로 (미토콘드리아 경로):
자극: DNA 손상, 산화 스트레스, ER 스트레스

1. 신호 전달:
   - p53 활성화 → PUMA, BAX, NOXA 유도
   - BAX/BAK 활성화 → 미토콘드리아 막 투과화

2. 미토콘드리아 사건:
   - Cytochrome c 방출
   - SMAC/DIABLO 방출 → IAP 억제
   - AIF 방출 → 핵 전위 → DNA 절편화

3. 카스파제 활성화:
   - Apoptosome 형성: Cytochrome c + Apaf1 + Procaspase-9
   - Caspase-9 활성화 → Caspase-3/7 활성화

외재적 경로 (데스 리셉터 경로):
자극: FasL, TNF, TRAIL

1. 리셉터 활성화:
   - Death receptor clustering
   - DISC 형성: FADD + Procaspase-8

2. 카스파제 활성화:
   - Caspase-8 활성화 → Caspase-3/7 활성화
   - Bid 절단 → tBid → 미토콘드리아 경로 증폭

조절 인자:
- 항세포사멸: Bcl-2, Bcl-xL, IAP
- 세포사멸 촉진: BAX, BAK, BIM, PUMA
- p53: DNA 손상 센서

세포사멸 vs 괴사:
- 세포사멸: 프로그램된 세포사멸, 염증 반응 없음
- 괴사: 병리적 세포사멸, 염증 반응 유발

질병:
- 암: 세포사멸 회피 (Bcl-2 과발현, p53 결함)
- 신경퇴행성 질환: 과도한 세포사멸
- 자가면역질환: 면역세포 세포사멸 결함
```

### 4.2 Cell Communication (세포 소통)

#### Gap junction - map04540
```
구조와 기능:
- Connexin 단백질로 구성
- 두 세포막을 관통하는 채널
- 분자량 1kDa 이하 분자 통과

Connexin 가족:
- Cx26: 청각, 피부
- Cx32: 간세포, 신경
- Cx43: 심근세포, 평활근세포

생리학적 역할:
1. 전기적 결합:
   - 심근: 활동전위 전파
   - 평활근: 수축 동조화
   - 신경: 전기적 시냅스

2. 대사적 결합:
   - 포도당, 아미노산 교환
   - Ca2+ 파동 전파
   - cAMP, IP3 확산

조절:
- 인산화: PKA, PKC에 의한 채널 폐쇄
- pH: 낮은 pH에서 폐쇄
- Ca2+: 높은 Ca2+에서 폐쇄

질병:
- 청력 소실: Cx26 돌연변이
- Charcot-Marie-Tooth: Cx32 돌연변이
- 백내장: Cx46, Cx50 돌연변이
- 심부정맥: Cx43 발현 감소
```

## 5. Organismal Systems (개체 시스템)

다세포 생물의 조직과 기관 수준의 생물학적 과정들입니다.

### 5.1 Immune System (면역계)

#### 적응 면역 반응 - map04612
```
T 세포 활성화:
1. 항원 인식:
   - TCR-MHC 상호작용
   - CD4+ T 세포: MHC class II
   - CD8+ T 세포: MHC class I

2. 보조 신호:
   - CD28-CD80/86: 활성화 신호
   - CTLA-4-CD80/86: 억제 신호
   - PD-1-PD-L1: 면역 체크포인트

3. 사이토카인:
   - IL-2: T 세포 증식
   - IFN-γ: Th1 분화
   - IL-4: Th2 분화
   - IL-17: Th17 분화

T 세포 아형:
- Th1: 세포내 병원체, IFN-γ, IL-2
- Th2: 기생충, 알레르기, IL-4, IL-5, IL-13
- Th17: 세균/진균, IL-17, IL-22
- Treg: 면역 관용, TGF-β, IL-10

B 세포 활성화:
1. 항원 인식: BCR-항원 결합
2. T 세포 도움: CD40-CD40L
3. 클래스 스위칭: AID 효소
4. 체세포 초돌연변이: 친화도 성숙
5. 플라즈마 세포 분화: 항체 분비

메모리 형성:
- 중심 메모리 T 세포 (TCM)
- 효과 메모리 T 세포 (TEM)  
- 메모리 B 세포
```

#### 보체 시스템 - map04610
```
3가지 활성화 경로:

1. 고전 경로 (Classical pathway):
   개시: 항원-항체 복합체
   C1q → C1r/C1s → C4 → C2 → C3

2. 렉틴 경로 (Lectin pathway):
   개시: 만노스 결합 렉틴 (MBL)
   MBL → MASP1/2 → C4 → C2 → C3

3. 대체 경로 (Alternative pathway):
   개시: C3 자발적 가수분해
   C3 → Factor B → Factor D → C3

공통 경로:
C3 convertase → C5 → C6,C7,C8,C9 → MAC

기능:
1. 용해: 막 공격 복합체 (MAC) 형성
2. 옵소닌화: C3b 침착 → 식세포 작용 촉진
3. 염증: C3a, C5a → 비만세포 탈과립
4. 면역 복합체 제거: CR1을 통한 제거

조절:
- C1 inhibitor: C1 억제
- Factor H: C3 convertase 억제
- CD55 (DAF): convertase 해체
- CD46 (MCP): C3b 분해 촉진
- CD59: MAC 형성 억제

질병:
- 유전성 혈관부종: C1 inhibitor 결핍
- 발작성 야간 혈색소뇨증: CD55, CD59 결핍
- 아토피형 용혈성 요독 증후군: Factor H 돌연변이
```

### 5.2 Endocrine System (내분비계)

#### 인슐린 신호전달 경로 - map04910
```
인슐린 신호 전달:
인슐린 → 인슐린 수용체 → IRS-1/2 → PI3K → Akt

인슐린 수용체:
- 구조: α2β2 테트라머
- α 서브유닛: 인슐린 결합
- β 서브유닛: 타이로신 키나제

IRS 단백질:
- IRS-1: 근육, 지방
- IRS-2: 간, 췌장 β세포
- 인산화 부위: 20개 이상 타이로신

대사 효과:
1. 포도당 대사:
   - Akt → AS160 → GLUT4 전위
   - Akt → GSK3β → 글리코겐 합성 증가
   - Akt → FoxO1 → 당신생합성 억제

2. 지질 대사:
   - Akt → mTORC1 → SREBP-1c → 지방산 합성
   - Akt → ACC → 지방산 합성 활성화
   - PDE3B → cAMP 감소 → HSL 불활성화

3. 단백질 합성:
   - Akt → mTORC1 → S6K1, 4E-BP1
   - 번역 개시 촉진

인슐린 저항성:
원인:
- 만성 염증: TNF-α, IL-6
- 지질독성: DAG, 세라미드
- ER 스트레스: JNK, IKK 활성화
- 산화 스트레스

기전:
- IRS-1 세린 인산화 → 분해 증가
- PI3K 활성 감소
- Akt 활성 감소

결과:
- 포도당 흡수 감소
- 당신생합성 증가
- 지방분해 증가
- 염증 반응 지속
```

## 6. Human Diseases (인간 질병)

질병과 관련된 분자 메커니즘을 나타내는 경로들입니다.

### 6.1 Cancer (암)

#### p53 신호전달 경로 - map04115
```
p53 활성화:
자극: DNA 손상, 산화 스트레스, 종양유전자 활성화

1. p53 안정화:
   - ATM/ATR → p53 Ser15 인산화
   - Chk1/Chk2 → p53 Ser20 인산화
   - MDM2 결합 차단 → p53 안정화

2. p53 활성화:
   - p300/CBP → p53 아세틸화
   - 전사 활성 증가

p53 표적 유전자:

1. 세포주기 정지:
   - p21 → CDK 억제 → G1/S 체크포인트
   - 14-3-3σ → CDC25C 격리 → G2/M 체크포인트

2. 세포사멸:
   - PUMA, NOXA → BAX/BAK 활성화
   - BAX → 미토콘드리아 투과화
   - APAF1 → 카스파제 활성화

3. DNA 수선:
   - DDB2 → NER 촉진
   - XPC → DNA 손상 인식
   - GADD45A → DNA 수선

4. 혈관신생 억제:
   - TSP1 → 혈관신생 억제
   - VEGF 발현 억제

5. 전이 억제:
   - MASPIN → 프로테아제 억제
   - CDH1 → 세포 접착 증가

p53 조절:
- MDM2: p53 유비퀴틴화 → 분해
- MDM4: p53 전사 활성 억제
- ARF: MDM2 억제 → p53 안정화

p53 돌연변이:
- 암의 50%에서 p53 돌연변이
- 핫스팟: 코돈 175, 248, 273, 282
- 기능 상실: DNA 결합 능력 소실
- 우성음성: 야생형 p53 억제
```

#### PI3K-Akt 경로와 암 - map05200
```
종양에서 PI3K-Akt 활성화:

1. 수용체 과발현:
   - EGFR: 폐암, 교모세포종
   - HER2: 유방암
   - PDGFR: 교모세포종

2. PI3K 활성화:
   - PIK3CA 돌연변이: 유방암, 대장암
   - PIK3R1 돌연변이: 교모세포종

3. PTEN 소실:
   - PTEN 삭제: 교모세포종 (80%)
   - PTEN 돌연변이: 자궁내막암

4. Akt 증폭:
   - AKT1: 유방암, 위암
   - AKT2: 췌장암, 난소암

종양에서 효과:

1. 세포 생존:
   - BAD, FoxO 인산화
   - p53 MDM2 경로 활성화
   - NF-κB 활성화

2. 세포 증식:
   - mTORC1 활성화 → 단백질 합성
   - GSK3β 억제 → c-Myc 안정화
   - p21, p27 인산화 → 분해

3. 혈관신생:
   - HIF-1α 안정화 → VEGF 발현
   - eNOS 활성화 → NO 생산

4. 전이:
   - EMT 촉진
   - 세포 운동성 증가

치료 표적:
- PI3K 억제제: Pictilisib, Buparlisib
- Akt 억제제: MK-2206, Ipatasertib
- mTOR 억제제: Rapamycin, Everolimus
- Dual PI3K/mTOR: Dactolisib, Gedatolisib
```

### 6.2 Metabolic Diseases (대사성 질환)

#### 제2형 당뇨병 - map04930
```
병태생리:

1. 인슐린 저항성:
   원인:
   - 복부 비만: 내장 지방 축적
   - 염증: TNF-α, IL-6 분비
   - 지질독성: FFA, DAG 증가
   - 미토콘드리아 기능 이상

   기전:
   - IRS-1 세린 인산화 증가
   - PI3K-Akt 신호 감소
   - GLUT4 전위 장애

2. β세포 기능 부전:
   원인:
   - 글루코독성: 고혈당에 의한 β세포 손상
   - 지질독성: 팔미트산에 의한 세포사멸
   - 아밀린 침착: IAPP 응집
   - 산화 스트레스: ROS 증가

   기전:
   - 인슐린 분비 감소
   - β세포 량 감소
   - Proinsulin 증가

합병증:

1. 미세혈관 합병증:
   - 당뇨병성 신장병: AGE 형성, PKC 활성화
   - 당뇨병성 망막병: VEGF 증가, 신생혈관 형성
   - 당뇨병성 신경병: 소르비톨 경로, myo-inositol 감소

2. 대혈관 합병증:
   - 관상동맥 질환: 죽상경화 촉진
   - 뇌졸중: 혈관 내피 기능 이상
   - 말초동맥 질환: 산화 스트레스 증가

치료:
- 메트포민: AMPK 활성화, 당신생합성 억제
- 설포닐우레아: K-ATP 채널 차단, 인슐린 분비 촉진
- DPP-4 억제제: GLP-1 분해 억제
- SGLT2 억제제: 신장 포도당 재흡수 억제
```

### 6.3 Neurodegenerative Diseases (신경퇴행성 질환)

#### 알츠하이머병 - map05010
```
병태생리:

1. 아밀로이드 경로:
   APP → β-secretase (BACE1) → sAPPβ + CTF99
   CTF99 → γ-secretase → AICD + Aβ40/42

   Aβ 응집:
   - 단량체 → 올리고머 → 원섬유 → 플라크
   - Aβ42: 더 응집하기 쉬움
   - 독성: 올리고머가 가장 독성

2. 타우 경로:
   - 타우 과인산화: GSK3β, CDK5, CK1
   - 미세소관 결합 능력 상실
   - 신경섬유다발 (NFT) 형성
   - 액손 수송 장애

3. 신경염증:
   - 미세아교세포 활성화: M1 표현형
   - 사이토카인 분비: TNF-α, IL-1β, IL-6
   - 보체 활성화: C1q, C3

4. 산화 스트레스:
   - 미토콘드리아 기능 이상
   - ROS 증가, 항산화 효소 감소
   - 지질 과산화, 단백질 산화

5. 시냅스 기능 이상:
   - NMDA 수용체 기능 이상
   - 아세틸콜린 감소: ChAT 활성 감소
   - 시냅스 소실: PSD-95, 시냅신 감소

위험 인자:
- APOE ε4: Aβ 클리어런스 감소
- TREM2: 미세아교세포 기능 이상
- CR1: 보체 조절 이상

치료:
- 아세틸콜린 에스터라제 억제제: 도네페질, 리바스티그민
- NMDA 길항제: 메만틴
- 아밀로이드 표적: 아두카누맙 (논란)
```

## 경로 간 상호작용과 네트워크 분석

### 크로스톡 (Crosstalk) 예시

#### mTOR을 중심으로 한 대사-성장 신호 통합
```
mTOR 복합체:

mTORC1 (mTOR + Raptor + mLST8):
활성화 인자:
- 인슐린/IGF-1 → PI3K-Akt → TSC1/2 억제
- 아미노산 → Rag GTPase → lysosomal recruitment
- 에너지: ATP/AMP ratio → AMPK 억제

억제 인자:
- AMPK → Raptor 인산화
- p53 → AMPK 활성화
- TSC1/2 → Rheb 불활성화

기능:
- S6K1 → ribosomal S6 → 번역 촉진
- 4E-BP1 → eIF4E 방출 → cap-dependent translation
- SREBP-1c → 지방산 합성
- PPARγ → 지방세포 분화

mTORC2 (mTOR + Rictor + mSIN1):
기능:
- Akt Ser473 인산화 → 완전 활성화
- SGK1 → 나트륨 수송
- PKCα → 세포골격 조절

질병:
- 암: mTOR 과활성화
- 당뇨병: mTOR에 의한 인슐린 저항성
- 신경퇴행성 질환: 자가포식 감소
- 노화: mTOR 신호 증가
```

### 시스템 수준 분석

#### 네트워크 의학적 접근
```python
def analyze_pathway_crosstalk():
    """경로 간 상호작용 네트워크 분석"""
    
    # 주요 허브 노드들
    hub_molecules = {
        'p53': ['DNA damage response', 'Apoptosis', 'Cell cycle'],
        'mTOR': ['PI3K-Akt', 'AMPK', 'Autophagy', 'Protein synthesis'],
        'NF-κB': ['TNF signaling', 'Toll-like receptor', 'Apoptosis'],
        'AMPK': ['Energy metabolism', 'mTOR signaling', 'Autophagy'],
        'β-catenin': ['Wnt signaling', 'Cell adhesion', 'Transcription']
    }
    
    # 질병에서의 네트워크 변화
    disease_perturbations = {
        'Cancer': {
            'upregulated': ['PI3K-Akt', 'Wnt', 'MAPK'],
            'downregulated': ['p53', 'Apoptosis', 'Cell cycle checkpoints']
        },
        'Diabetes': {
            'upregulated': ['mTOR', 'Inflammatory signaling'],
            'downregulated': ['Insulin signaling', 'AMPK']
        },
        'Alzheimer': {
            'upregulated': ['Neuroinflammation', 'Oxidative stress'],
            'downregulated': ['Synaptic signaling', 'Autophagy']
        }
    }
    
    return {
        'network_hubs': hub_molecules,
        'disease_signatures': disease_perturbations
    }
```

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "KEGG PATHWAY \ubd84\ub958 \uccb4\uacc4 \uc815\ub9ac", "status": "completed"}, {"content": "\uc8fc\uc694 \ub300\uc0ac \uacbd\ub85c\ub4e4 \uc0c1\uc138 \uc124\uba85", "status": "completed"}, {"content": "\uc2e0\ud638\uc804\ub2ec \uacbd\ub85c\ub4e4 \ubd84\uc11d", "status": "completed"}, {"content": "\uc9c8\ubcd1 \uad00\ub828 \uacbd\ub85c\ub4e4 \uc815\ub9ac", "status": "completed"}, {"content": "\uacbd\ub85c \uac04 \uc0c1\ud638\uc791\uc6a9 \ubd84\uc11d", "status": "completed"}]