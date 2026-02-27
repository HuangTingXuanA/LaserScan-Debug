# DP算法测试场景

## 场景1：前面有干扰线

### 输入
```
切片序列：S0  S1  S2  S3  S4  S5  S6  S7  S8  S9
原始线：  L0  L0  L0  L1  L1  L1  L2  L2  L2  L2
Cost矩阵：
  S0: [INV, INV, INV, INV, INV, INV, INV]  ← 干扰线
  S1: [INV, INV, INV, INV, INV, INV, INV]  ← 干扰线
  S2: [INV, INV, INV, INV, INV, INV, INV]  ← 干扰线
  S3: [0.5, INV, INV, INV, INV, INV, INV]  ← 第一个有效切片
  S4: [0.6, INV, INV, INV, INV, INV, INV]
  S5: [0.7, INV, INV, INV, INV, INV, INV]
  S6: [INV, 0.8, INV, INV, INV, INV, INV]
  S7: [INV, 0.9, INV, INV, INV, INV, INV]
  S8: [INV, 1.0, INV, INV, INV, INV, INV]
  S9: [INV, 1.1, INV, INV, INV, INV, INV]
```

### DP过程
```
初始化 (i=0):
  dp[0][k] = 0.0 for all k (跳过S0)
  backtrack[0][k] = {false, -1, -1}

i=1 (S1):
  dp[1][k] = 0.0 for all k (继承，跳过S1)
  backtrack[1][k] = {false, 0, k}

i=2 (S2):
  dp[2][k] = 0.0 for all k (继承，跳过S2)
  backtrack[2][k] = {false, 1, k}

i=3 (S3, L1):
  匹配到P0: dp[3][0] = 0.0 + 0.0 + 0.5 = 0.5
  backtrack[3][0] = {true, 0, 0} (从初始状态转移)

i=4 (S4, L1):
  匹配到P0: dp[4][0] = 0.5 + 0.0 + 0.6 = 1.1
  backtrack[4][0] = {true, 3, 0} (同线同平面)

i=5 (S5, L1):
  匹配到P0: dp[5][0] = 1.1 + 0.0 + 0.7 = 1.8
  backtrack[5][0] = {true, 4, 0}

i=6 (S6, L2):
  匹配到P1: dp[6][1] = 1.8 + 2.0 + 0.8 = 4.6
  backtrack[6][1] = {true, 5, 0} (不同线异平面，j=0<k=1)

...
```

### 预期结果
```
匹配路径：S3(P0) → S4(P0) → S5(P0) → S6(P1) → S7(P1) → S8(P1) → S9(P1)
跳过切片：S0, S1, S2 (干扰线)
光平面序列：P0, P0, P0, P1, P1, P1, P1 ✓ 单调不减
```

## 场景2：中间有遮挡

### 输入
```
切片序列：S0  S1  S2  S3  S4  S5  S6  S7  S8
原始线：  L1  L1  L1  L2  L2  L2  L3  L3  L3
Cost矩阵：
  S0: [0.5, INV, INV, INV, INV, INV, INV]
  S1: [0.6, INV, INV, INV, INV, INV, INV]
  S2: [INV, 0.7, INV, INV, INV, INV, INV]
  S3: [INV, 0.8, INV, INV, INV, INV, INV]
  S4: [INV, INV, INV, INV, INV, INV, INV]  ← 遮挡
  S5: [INV, INV, INV, INV, INV, INV, INV]  ← 遮挡
  S6: [INV, INV, 0.9, INV, INV, INV, INV]
  S7: [INV, INV, 1.0, INV, INV, INV, INV]
  S8: [INV, INV, 1.1, INV, INV, INV, INV]
```

### DP过程
```
i=0-3: 正常匹配 P0 → P1

i=4 (S4, L2):
  所有平面cost=INV
  跳过: dp[4][k] = dp[3][k] for k>=1
  backtrack[4][k] = {false, 3, k}

i=5 (S5, L2):
  所有平面cost=INV
  跳过: dp[5][k] = dp[4][k] for k>=1
  backtrack[5][k] = {false, 4, k}

i=6 (S6, L3):
  匹配到P2: 
    从S3(P1)转移: dp[6][2] = dp[3][1] + skip_penalty(2) + transition + 0.9
                            = cost_S3 + 2.0 + 2.0 + 0.9
  backtrack[6][2] = {true, 3, 1}
```

### 预期结果
```
匹配路径：S0(P0) → S1(P0) → S2(P1) → S3(P1) → [跳过S4,S5] → S6(P2) → S7(P2) → S8(P2)
跳过切片：S4, S5 (遮挡)
光平面序列：P0, P0, P1, P1, P2, P2, P2 ✓ 单调不减
```

## 场景3：假连通（同一线跨越多个平面）

### 输入
```
切片序列：S0  S1  S2  S3  S4  S5
原始线：  L1  L1  L1  L1  L1  L1  ← 同一条线
Cost矩阵：
  S0: [0.5, INV, INV, INV]
  S1: [0.6, INV, INV, INV]
  S2: [0.7, 0.8, INV, INV]  ← 可以匹配P0或P1
  S3: [INV, 0.9, INV, INV]
  S4: [INV, 1.0, INV, INV]
  S5: [INV, INV, 1.1, INV]
```

### DP过程
```
i=0-1: 匹配到P0

i=2 (S2, L1):
  选项1: 匹配到P0
    dp[2][0] = dp[1][0] + 0.0 + 0.7 = 1.1 + 0.7 = 1.8
  选项2: 匹配到P1
    dp[2][1] = dp[1][0] + 5.0 + 0.8 = 1.1 + 5.0 + 0.8 = 6.9
    (同线切换平面，惩罚5.0)
  
  选择P0 (代价更低)

i=3 (S3, L1):
  只能匹配P1
  dp[3][1] = dp[2][0] + 5.0 + 0.9 = 1.8 + 5.0 + 0.9 = 7.7
  (同线切换平面)

i=4-5: 继续匹配P1和P2
```

### 预期结果
```
匹配路径：S0(P0) → S1(P0) → S2(P0) → S3(P1) → S4(P1) → S5(P2)
光平面序列：P0, P0, P0, P1, P1, P2 ✓ 单调不减
切换点：S2→S3 (同线切换平面，有惩罚但允许)
```

## 场景4：x距离约束

### 输入
```
切片序列：S0  S1  S2  S3
原始线：  L1  L1  L2  L2
中心x：   10  12  30  32  ← S0-S1和S2-S3相距很远
Cost矩阵：
  S0: [0.5, INV, INV]
  S1: [0.6, INV, INV]
  S2: [0.7, INV, INV]  ← 也能匹配P0
  S3: [0.8, INV, INV]
```

### DP过程
```
i=0-1: L1匹配到P0

i=2 (S2, L2):
  尝试匹配到P0:
    dx = 30 - 12 = 18 > 15
    不同线且x相差太远
    transition_penalty = 1000.0
    dp[2][0] = 1.1 + 1000.0 + 0.7 = 1001.8 (代价极高)
  
  尝试匹配到P1:
    dp[2][1] = 1.1 + 2.0 + INV (P1无效)
  
  最优：跳过S2或接受高代价

i=3 (S3, L2):
  类似S2
```

### 预期结果
```
如果有其他平面可选：
  S0-S1匹配P0，S2-S3匹配P1或更高平面
  
如果只有P0可选：
  可能跳过S2-S3（代价太高）
  或者接受高代价（如果没有更好选择）
```

## 关键测试点

1. **干扰线处理**：前面的无效切片能否正确跳过
2. **遮挡处理**：中间的无效切片能否正确跳过
3. **光平面单调性**：是否严格满足j <= k
4. **假连通惩罚**：同线切换平面是否有合理惩罚
5. **x距离约束**：不同线x相差太远是否禁止同平面
6. **路径完整性**：回溯是否能正确重建路径
7. **跳过标记**：backtrack.matched是否正确标记
