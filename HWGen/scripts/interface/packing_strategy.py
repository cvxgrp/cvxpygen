"""
Interface Packing Strategy with 2-Tier Adaptive Approach
Tier 1: Spatial Multiplexing (default, best performance, full parallel)
Tier 2: Time Division Multiplexing (IOB-constrained, configurable batching)
"""

import math


class PackingStrategy:
    """Decide input/output port packing strategy with IOB awareness"""
    
    # ========== Tier 1: Spatial Multiplexing Configuration ==========
    # Preferred port widths (standard AXI widths)
    SPATIAL_PREFERRED_WIDTHS = [32, 64, 128, 256, 512]
    
    # Default preferred widths for Tier 1
    TIER1_INPUT_PREFERRED = 64      # Default input port width
    TIER1_OUTPUT_PREFERRED = 128    # Default output port width
    
    # ========== Tier 2: TDM Configuration ==========
    TDM_BATCH_SEARCH_MIN = 2        # Minimum batch size
    TDM_BATCH_SEARCH_MAX = 64       # Maximum batch size
    TDM_BATCH_ID_WIDTH = 8          # Supports up to 256 batches
    
    # ========== Port Width Limits ==========
    MAX_WIDE_PORT = 512             # AXI4-Stream maximum
    
    # IOB estimation factors
    IOB_PER_CTRL_PIN = 1
    IOB_MARGIN = 1.15               # 15% safety margin (考虑 TDM 控制开销)
    
    def __init__(self, n_params, n_sols, data_width, max_iob=None):
        """
        Args:
            n_params: Number of input parameters
            n_sols: Number of output solutions
            data_width: Data width in bits (per parameter/solution)
            max_iob: Maximum IOB pin count (None = unlimited, use Tier 1)
        """
        self.n_params = n_params
        self.n_sols = n_sols
        self.data_width = data_width
        self.max_iob = max_iob
        
        self.input_strategy = None
        self.output_strategy = None
        self.tier_used = None
        
        self._decide()
    
    def _decide(self):
        """Decide packing strategy based on IOB constraints"""
        if self.max_iob is None:
            # 无约束：使用 Tier 1（空间复用，最佳性能）
            self._decide_tier1_spatial()
            self.tier_used = 1
        else:
            # 有约束：先尝试 Tier 1，失败则用 Tier 2
            self._decide_with_constraint()
    
    def _decide_tier1_spatial(self):
        """Tier 1: Spatial Multiplexing - 自动选择最优端口宽度"""
        total_input_bits = self.n_params * self.data_width
        total_output_bits = self.n_sols * self.data_width
        
        # 输入端口策略
        self.input_strategy = self._pack_spatial(
            total_bits=total_input_bits,
            n_items=self.n_params,
            preferred_width=self.TIER1_INPUT_PREFERRED,
            item_name='params'
        )
        
        # 输出端口策略
        self.output_strategy = self._pack_spatial(
            total_bits=total_output_bits,
            n_items=self.n_sols,
            preferred_width=self.TIER1_OUTPUT_PREFERRED,
            item_name='sols'
        )
    
    def _pack_spatial(self, total_bits, n_items, preferred_width, item_name):
        """
        空间复用打包：自动选择最优端口宽度
        
        选择原则：
        1. 优先使用标准 AXI 宽度（32/64/128/256/512）
        2. 最小化端口数量
        3. 避免浪费（端口利用率 > 50%）
        
        Args:
            total_bits: Total bit width needed
            n_items: Number of items (params or solutions)
            preferred_width: Preferred port width hint
            item_name: 'params' or 'sols' for dictionary key
        
        Returns:
            Strategy dictionary with mode='multi-parallel' or 'packed'
        """
        # 单端口即可容纳
        if total_bits <= max(self.SPATIAL_PREFERRED_WIDTHS):
            # 选择刚好能容纳的最小标准宽度
            for width in self.SPATIAL_PREFERRED_WIDTHS:
                if total_bits <= width:
                    return {
                        'mode': 'packed',
                        'num_ports': 1,
                        'port_width': total_bits,  # 实际使用宽度
                        f'{item_name}_per_port': [n_items],
                        'utilization': total_bits / width
                    }
        
        # 需要多端口：选择最优宽度组合
        best_config = None
        best_score = float('inf')
        
        # 从 preferred_width 开始尝试，然后尝试其他宽度
        widths_to_try = sorted(
            self.SPATIAL_PREFERRED_WIDTHS,
            key=lambda w: abs(w - preferred_width)
        )
        
        for port_width in widths_to_try:
            items_per_port = port_width // self.data_width
            if items_per_port == 0:
                continue
            
            num_ports = math.ceil(n_items / items_per_port)
            
            # 计算端口利用率
            total_capacity = num_ports * port_width
            utilization = total_bits / total_capacity
            
            # 评分函数：优先选择端口少且利用率高的
            # 利用率低于 50% 惩罚
            penalty = 0 if utilization >= 0.5 else (0.5 - utilization) * 100
            score = num_ports + penalty
            
            if score < best_score:
                best_score = score
                
                # 分配数据到各端口
                distribution = []
                remaining = n_items
                for i in range(num_ports):
                    n = min(items_per_port, remaining)
                    distribution.append(n)
                    remaining -= n
                
                best_config = {
                    'mode': 'multi-parallel',
                    'num_ports': num_ports,
                    'port_width': port_width,
                    f'{item_name}_per_port': distribution,
                    'utilization': utilization
                }
        
        if best_config is None:
            raise ValueError(
                f"Cannot pack {n_items} items of {self.data_width}-bit width "
                f"into standard port widths. Total bits: {total_bits}"
            )
        
        return best_config
    
    def _decide_tier2_tdm(self, target_iob):
        """
        Tier 2: Time Division Multiplexing
        自动搜索满足 IOB 约束的最优批次大小
        
        Args:
            target_iob: 目标 IOB 数量
        """
        # 搜索最优批次大小
        input_batch = self._find_optimal_batch_size(
            n_items=self.n_params,
            target_iob=target_iob,
            is_input=True
        )
        
        output_batch = self._find_optimal_batch_size(
            n_items=self.n_sols,
            target_iob=target_iob,
            is_input=False
        )
        
        # 验证 IOB 约束
        estimated_iob = self._estimate_tdm_iob(input_batch, output_batch)
        
        if estimated_iob > target_iob:
            raise ValueError(
                f"Cannot fit design within IOB constraint {target_iob}. "
                f"Minimum achievable with TDM: {estimated_iob}. "
                f"Consider: 1) Increasing max_iob, 2) Reducing parameters/solutions."
            )
        
        # 构建策略
        input_num_batches = math.ceil(self.n_params / input_batch)
        output_num_batches = math.ceil(self.n_sols / output_batch)
        
        self.input_strategy = {
            'mode': 'tdm',
            'num_ports': 1,  # TDM 使用单端口
            'port_width': input_batch * self.data_width,
            'params_per_port': [self.n_params],  # 兼容旧接口
            'batch_size': input_batch,
            'num_batches': input_num_batches,
            'batch_id_width': self.TDM_BATCH_ID_WIDTH,
            'total_items': self.n_params
        }
        
        self.output_strategy = {
            'mode': 'tdm',
            'num_ports': 1,  # TDM 使用单端口
            'port_width': output_batch * self.data_width,
            'sols_per_port': [self.n_sols],  # 兼容旧接口
            'batch_size': output_batch,
            'num_batches': output_num_batches,
            'batch_id_width': self.TDM_BATCH_ID_WIDTH,
            'total_items': self.n_sols
        }
    
    def _find_optimal_batch_size(self, n_items, target_iob, is_input):
        """
        搜索最优批次大小
        
        目标：在满足 IOB 约束的前提下，最大化批次大小（减少传输次数）
        
        Args:
            n_items: 项目数量
            target_iob: 目标 IOB 限制
            is_input: True=输入, False=输出
        
        Returns:
            最优批次大小
        """
        min_batch = self.TDM_BATCH_SEARCH_MIN
        max_batch = min(n_items, self.TDM_BATCH_SEARCH_MAX)
        
        best_batch = min_batch
        
        for batch_size in range(min_batch, max_batch + 1):
            # 临时估算 IOB
            if is_input:
                test_input_batch = batch_size
                # 如果输出策略已决定，使用其 batch_size；否则用保守估计
                if self.output_strategy and 'batch_size' in self.output_strategy:
                    test_output_batch = self.output_strategy['batch_size']
                else:
                    test_output_batch = min_batch
            else:
                # 输入策略应该已决定
                if self.input_strategy and 'batch_size' in self.input_strategy:
                    test_input_batch = self.input_strategy['batch_size']
                else:
                    test_input_batch = min_batch
                test_output_batch = batch_size
            
            estimated = self._estimate_tdm_iob(test_input_batch, test_output_batch)
            
            if estimated <= target_iob:
                best_batch = batch_size
            else:
                break  # 超出约束，停止增加
        
        return best_batch
    
    def _estimate_tdm_iob(self, input_batch_size, output_batch_size):
        """估算 TDM 配置的 IOB 数量"""
        # 输入 IOB
        input_data_iob = input_batch_size * self.data_width
        input_batch_id_iob = self.TDM_BATCH_ID_WIDTH
        input_ctrl_iob = 2 * self.IOB_PER_CTRL_PIN  # valid + ready
        input_total = input_data_iob + input_batch_id_iob + input_ctrl_iob
        
        # 输出 IOB
        output_data_iob = output_batch_size * self.data_width
        output_batch_id_iob = self.TDM_BATCH_ID_WIDTH
        output_ctrl_iob = 2 * self.IOB_PER_CTRL_PIN  # valid + ready
        output_total = output_data_iob + output_batch_id_iob + output_ctrl_iob
        
        # 其他信号（时钟、复位）
        misc_iob = 2
        
        total = input_total + output_total + misc_iob
        
        return int(total * self.IOB_MARGIN)
    
    def _decide_with_constraint(self):
        """在 IOB 约束下决策"""
        # 先尝试 Tier 1（空间复用）
        self._decide_tier1_spatial()
        tier1_iob = self._estimate_current_iob()
        
        if tier1_iob <= self.max_iob:
            self.tier_used = 1
            return
        
        # Tier 1 超出约束，使用 Tier 2（TDM）
        self._decide_tier2_tdm(self.max_iob)
        tier2_iob = self._estimate_current_iob()
        
        if tier2_iob > self.max_iob:
            # TDM 仍然超出约束（极端情况）
            raise ValueError(
                f"Cannot fit design within IOB constraint {self.max_iob}. "
                f"Even TDM configuration needs {tier2_iob} IOBs. "
                f"Consider: 1) Increasing max_iob, 2) Reducing parameters/solutions, "
                f"3) Using external multiplexing."
            )
        
        self.tier_used = 2
    
    def _estimate_current_iob(self):
        """估算当前策略的 IOB 数量"""
        input_mode = self.input_strategy['mode']
        output_mode = self.output_strategy['mode']
        
        # 输入 IOB
        if input_mode == 'tdm':
            input_iob = (self.input_strategy['port_width'] + 
                        self.input_strategy['batch_id_width'] +
                        2 * self.IOB_PER_CTRL_PIN)
        else:  # spatial (packed 或 multi-parallel)
            num_ports = self.input_strategy['num_ports']
            port_width = self.input_strategy['port_width']
            input_iob = num_ports * port_width + 2 * self.IOB_PER_CTRL_PIN
        
        # 输出 IOB
        if output_mode == 'tdm':
            output_iob = (self.output_strategy['port_width'] + 
                         self.output_strategy['batch_id_width'] +
                         2 * self.IOB_PER_CTRL_PIN)
        else:  # spatial (packed 或 multi-parallel)
            num_ports = self.output_strategy['num_ports']
            port_width = self.output_strategy['port_width']
            output_iob = num_ports * port_width + 2 * self.IOB_PER_CTRL_PIN
        
        # 其他信号
        misc_iob = 2  # clk + rst_n
        
        total = input_iob + output_iob + misc_iob
        
        return int(total * self.IOB_MARGIN)
    
    def get_input_port_name(self, index):
        """获取输入端口名称（保持向后兼容）"""
        mode = self.input_strategy['mode']
        num_ports = self.input_strategy.get('num_ports', 1)
        
        if mode in ['tdm', 'packed'] or num_ports == 1:
            return 's_axis_tdata'
        else:
            return f's_axis_tdata_{index}'
    
    def get_output_port_name(self, index):
        """获取输出端口名称（保持向后兼容）"""
        mode = self.output_strategy['mode']
        num_ports = self.output_strategy.get('num_ports', 1)
        
        if mode in ['tdm', 'packed'] or num_ports == 1:
            return 'm_axis_tdata'
        else:
            return f'm_axis_tdata_{index}'
    
    def get_summary(self):
        """生成人类可读的策略摘要"""
        tier_names = {
            1: 'Spatial Multiplexing (Full Parallel)',
            2: 'Time Division Multiplexing (IOB Optimized)'
        }
        
        summary = []
        summary.append(f"Packing Strategy: Tier {self.tier_used} - {tier_names[self.tier_used]}")
        summary.append("")
        
        # 输入策略
        summary.append("Input Configuration:")
        if self.input_strategy['mode'] in ['packed', 'multi-parallel']:
            mode_name = 'Spatial Multiplexing'
            summary.append(f"  Mode: {mode_name}")
            summary.append(f"  Ports: {self.input_strategy['num_ports']}")
            summary.append(f"  Port Width: {self.input_strategy['port_width']} bits")
            summary.append(f"  Utilization: {self.input_strategy.get('utilization', 1.0)*100:.1f}%")
            summary.append(f"  Distribution: {self.input_strategy['params_per_port']}")
        else:  # tdm
            summary.append(f"  Mode: Time Division Multiplexing")
            summary.append(f"  Total Parameters: {self.input_strategy['total_items']}")
            summary.append(f"  Batch Size: {self.input_strategy['batch_size']} params/batch")
            summary.append(f"  Num Batches: {self.input_strategy['num_batches']}")
            summary.append(f"  Port Width: {self.input_strategy['port_width']} bits")
            summary.append(f"  Batch ID Width: {self.input_strategy['batch_id_width']} bits")
        
        summary.append("")
        
        # 输出策略
        summary.append("Output Configuration:")
        if self.output_strategy['mode'] in ['packed', 'multi-parallel']:
            mode_name = 'Spatial Multiplexing'
            summary.append(f"  Mode: {mode_name}")
            summary.append(f"  Ports: {self.output_strategy['num_ports']}")
            summary.append(f"  Port Width: {self.output_strategy['port_width']} bits")
            summary.append(f"  Utilization: {self.output_strategy.get('utilization', 1.0)*100:.1f}%")
            summary.append(f"  Distribution: {self.output_strategy['sols_per_port']}")
        else:  # tdm
            summary.append(f"  Mode: Time Division Multiplexing")
            summary.append(f"  Total Solutions: {self.output_strategy['total_items']}")
            summary.append(f"  Batch Size: {self.output_strategy['batch_size']} sols/batch")
            summary.append(f"  Num Batches: {self.output_strategy['num_batches']}")
            summary.append(f"  Port Width: {self.output_strategy['port_width']} bits")
            summary.append(f"  Batch ID Width: {self.output_strategy['batch_id_width']} bits")
        
        summary.append("")
        
        # IOB 统计
        estimated_iob = self._estimate_current_iob()
        if self.max_iob:
            utilization = (estimated_iob / self.max_iob) * 100
            summary.append(f"IOB Usage: {estimated_iob} / {self.max_iob} ({utilization:.1f}%)")
            
            if self.tier_used == 2:
                # 计算 Tier 1 的 IOB（对比）
                # 简化估算：假设 Tier 1 使用全部数据位
                tier1_data = (self.n_params * self.data_width + 
                             self.n_sols * self.data_width)
                tier1_ctrl = 4  # input/output valid+ready
                tier1_misc = 2  # clk+rst
                tier1_iob = int((tier1_data + tier1_ctrl + tier1_misc) * self.IOB_MARGIN)
                
                savings = tier1_iob - estimated_iob
                savings_pct = (savings / tier1_iob) * 100 if tier1_iob > 0 else 0
                summary.append(f"IOB Savings: {savings} pins ({savings_pct:.1f}% reduction from Tier 1)")
        else:
            summary.append(f"IOB Usage: {estimated_iob} (no constraint)")
        
        return '\n'.join(summary)
    
    def __repr__(self):
        return (f"PackingStrategy(Tier {self.tier_used}, "
                f"Input: {self.input_strategy['mode']}, "
                f"Output: {self.output_strategy['mode']})")