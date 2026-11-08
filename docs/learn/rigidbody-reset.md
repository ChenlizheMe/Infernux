# 刚体回合重置

`Physics.reset_rigidbodies` 在两个已完成的物理步之间，一次恢复一组刚体的世界位姿、线速度和角速度，同时清除它们的求解历史。适用于强化学习回合、角色复位、机械装置重置。

```python
from infernux.physics import Physics

fields = ('position', 'rotation', 'linear_velocity', 'angular_velocity')

# 在 Play 中、碰撞体已经准备好后保存初始状态。
initial = Physics.get_rigidbody_states(bodies, fields=fields)
initial['linear_velocity'].fill(0)
initial['angular_velocity'].fill(0)

# 先完成本回合的观察、奖励和最终 value，再重置。
# 已发出的推理请求必须完成或取消，旧回合动作必须丢弃。
Physics.reset_rigidbodies(bodies, initial)
```

输入字典恰好包含以下四项。数据为连续的 float32 CPU NumPy 数组或 CPU Buffer，行顺序与 `bodies` 一致，四元数使用 xyzw。

| 字段 | 形状 | 单位 |
| --- | --- | --- |
| position | N × 3 | 世界坐标，米 |
| rotation | N × 4 | 世界旋转，非零四元数会归一化 |
| linear_velocity | N × 3 | 世界线速度，米/秒 |
| angular_velocity | N × 3 | 世界角速度，弧度/秒 |

一次调用的行为：

- 在任何对象改变之前检查全部输入、对象是否驻留以及重复对象。
- 清除累积力、力矩、未执行的运动目标和相关关节的 warm start。
- 丢弃相关接触的旧几何缓存、冲量与事件身份；重置后仍有接触时产生新 Enter，旧回合 Exit 不进入新回合。
- 保留刚体、形状和关节身份，以及质量、惯量、限位、电机目标和碰撞过滤规则。需要更改目标时由调用者明确设置。
- 一起更新物理位姿、显示与插值缓存。传入父子刚体的顺序不影响结果。
- 唤醒相关刚体，不改变已完成步数、模拟时间或 Play 会话。

调用必须位于物理 owner 线程、Play 中的完整物理步之间；固定更新和碰撞回调中不能直接重置。回调可以记录结束请求，由应用在本步完成后处理。

世界的完成步数并不能区分“同一步内重置前后”的动作。环境应保留自己的回合身份，并在重置时丢弃上一回合的观察和动作请求。模型 `.inm` 检查点保存模型与训练状态；场景回合状态由应用管理。
