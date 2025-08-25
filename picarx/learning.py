from __future__ import annotations

import random
import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
import os

from .logging_setup import init_logger


@dataclass
class BanditConfig:
    eps: float = 0.1
    arms: List[float] = None  # speed scale candidates
    decay_rate: float = 0.995  # epsilon decay for exploration->exploitation
    min_eps: float = 0.02  # minimum exploration rate
    context_window: int = 50  # window for context-aware learning
    adaptation_rate: float = 0.1  # how fast to adapt to changing conditions

    def __post_init__(self):
        if self.arms is None:
            self.arms = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]  # More aggressive options


@dataclass
class ContextState:
    """Environment context for adaptive learning"""
    battery_level: float = 1.0
    obstacle_density: float = 0.0  # 0=clear, 1=very cluttered
    lighting_quality: float = 1.0  # 0=poor, 1=excellent
    surface_type: str = "unknown"  # smooth, rough, mixed
    time_of_day: str = "day"  # day, night, dawn, dusk
    recent_collisions: int = 0
    successful_maneuvers: int = 0


@dataclass
class PerformanceMetrics:
    """Multi-objective performance tracking"""
    speed_efficiency: float = 0.0
    safety_score: float = 0.0
    exploration_score: float = 0.0
    energy_efficiency: float = 0.0
    mission_progress: float = 0.0
    timestamp: float = field(default_factory=time.time)


class AdaptiveBandit:
    """Enhanced multi-armed bandit with context awareness and self-improvement"""
    
    def __init__(self, cfg: BanditConfig) -> None:
        self.cfg = cfg
        self.log = init_logger("picarx.learning")
        
        # Basic bandit state
        self.value: Dict[float, float] = {a: 0.0 for a in cfg.arms}
        self.count: Dict[float, int] = {a: 0 for a in cfg.arms}
        self.confidence: Dict[float, float] = {a: 0.0 for a in cfg.arms}
        
        # Adaptive learning
        self.current_eps = cfg.eps
        self.context_history: deque = deque(maxlen=cfg.context_window)
        self.performance_history: deque = deque(maxlen=100)
        self.arm_context_performance: Dict[float, Dict[str, List[float]]] = {
            arm: {} for arm in cfg.arms
        }
        
        # Self-improvement metrics
        self.learning_rate = 0.1
        self.session_start = time.time()
        self.total_actions = 0
        self.successful_actions = 0
        self.adaptation_triggers = 0
        
        # Rate limiting for stability
        self.last_update_time = 0.0
        self.min_update_interval = 2.0  # Minimum 2 seconds between updates
        self.last_adaptation_time = 0.0
        self.min_adaptation_interval = 10.0  # Minimum 10 seconds between adaptations
        self.last_log_time = 0.0
        self.log_interval = 5.0  # Only log significant events every 5 seconds
        self.max_exploration = 0.15  # Cap exploration at 15%
        self.exploration_step = 0.02  # Smaller exploration adjustments
        
        # Persistence
        self.save_path = os.path.expanduser("~/.picarx/learning_state.json")
        self._load_state()
        
    def select(self, context: Optional[ContextState] = None) -> float:
        """Select arm with context-aware adaptive strategy"""
        self.total_actions += 1
        
        # Update exploration rate
        self._update_exploration_rate()
        
        # Context-aware selection
        if context and len(self.context_history) > 10:
            return self._context_aware_select(context)
        
        # UCB1 with adaptive confidence
        if random.random() < self.current_eps:
            # Intelligent exploration: prefer less-tried arms in current context
            unexplored = [a for a in self.cfg.arms if self.count[a] < 3]
            if unexplored:
                arm = random.choice(unexplored)
                self.log.debug(f"explore unexplored arm={arm}")
                return arm
            else:
                arm = random.choice(self.cfg.arms)
                self.log.debug(f"explore random arm={arm}")
                return arm
        
        # Exploit with Upper Confidence Bound
        total_count = sum(self.count.values()) + 1
        ucb_values = {}
        
        for arm in self.cfg.arms:
            if self.count[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                confidence_bonus = math.sqrt(2 * math.log(total_count) / self.count[arm])
                ucb_values[arm] = self.value[arm] + confidence_bonus
        
        best_arm = max(ucb_values.keys(), key=lambda k: ucb_values[k])
        self.log.debug(f"exploit arm={best_arm} (UCB={ucb_values[best_arm]:.3f})")
        return best_arm
    
    def _context_aware_select(self, context: ContextState) -> float:
        """Select arm based on context similarity"""
        context_key = self._context_to_key(context)
        
        # Find arms that performed well in similar contexts
        context_scores = {}
        for arm in self.cfg.arms:
            if context_key in self.arm_context_performance[arm]:
                recent_scores = self.arm_context_performance[arm][context_key][-5:]
                if recent_scores:
                    context_scores[arm] = sum(recent_scores) / len(recent_scores)
                else:
                    context_scores[arm] = 0.0
            else:
                context_scores[arm] = 0.0
        
        # Blend context-aware and UCB selection
        if max(context_scores.values()) > 0.1:
            best_context_arm = max(context_scores.keys(), key=lambda k: context_scores[k])
            self.log.debug(f"context-aware select arm={best_context_arm}")
            return best_context_arm
        
        # Fall back to normal UCB
        return self._ucb_select()
    
    def _ucb_select(self) -> float:
        """Upper Confidence Bound selection"""
        total_count = sum(self.count.values()) + 1
        ucb_values = {}
        
        for arm in self.cfg.arms:
            if self.count[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                confidence_bonus = math.sqrt(2 * math.log(total_count) / self.count[arm])
                ucb_values[arm] = self.value[arm] + confidence_bonus
        
        return max(ucb_values.keys(), key=lambda k: ucb_values[k])
    
    def update(self, arm: float, reward: float, context: Optional[ContextState] = None, 
               metrics: Optional[PerformanceMetrics] = None) -> None:
        """Update with multi-objective reward and context - rate limited for stability"""
        
        current_time = time.time()
        
        # Rate limit updates to prevent spam
        if current_time - self.last_update_time < self.min_update_interval:
            return
        
        self.last_update_time = current_time
        
        # Basic bandit update with adaptive learning rate
        n = self.count[arm] + 1
        v = self.value[arm]
        
        # Adaptive learning rate based on confidence and recency
        adaptive_lr = self.learning_rate / (1 + self.count[arm] * 0.01)
        
        self.count[arm] = n
        self.value[arm] = v + adaptive_lr * (reward - v)
        
        # Update confidence based on consistency
        if n > 1:
            prediction_error = abs(reward - v)
            self.confidence[arm] = 0.9 * self.confidence[arm] + 0.1 * (1.0 - prediction_error)
        
        # Context-aware learning
        if context:
            self._update_context_performance(arm, reward, context)
            self.context_history.append(context)
        
        # Multi-objective learning
        if metrics:
            self.performance_history.append(metrics)
            self._update_multi_objective_learning(arm, metrics)
        
        # Track success rate
        if reward > 0.1:
            self.successful_actions += 1
        
        # Trigger adaptation if needed (rate limited)
        if current_time - self.last_adaptation_time > self.min_adaptation_interval:
            self._check_adaptation_triggers()
        
        # Periodic state saving
        if self.total_actions % 50 == 0:
            self._save_state()
    
    def _update_context_performance(self, arm: float, reward: float, context: ContextState) -> None:
        """Track performance in different contexts"""
        context_key = self._context_to_key(context)
        
        if context_key not in self.arm_context_performance[arm]:
            self.arm_context_performance[arm][context_key] = []
        
        self.arm_context_performance[arm][context_key].append(reward)
        
        # Keep only recent performance data
        if len(self.arm_context_performance[arm][context_key]) > 20:
            self.arm_context_performance[arm][context_key] = \
                self.arm_context_performance[arm][context_key][-15:]
    
    def _context_to_key(self, context: ContextState) -> str:
        """Convert context to hashable key"""
        battery_bucket = "high" if context.battery_level > 0.7 else "med" if context.battery_level > 0.3 else "low"
        obstacle_bucket = "clear" if context.obstacle_density < 0.3 else "med" if context.obstacle_density < 0.7 else "dense"
        light_bucket = "good" if context.lighting_quality > 0.7 else "med" if context.lighting_quality > 0.4 else "poor"
        
        return f"{battery_bucket}_{obstacle_bucket}_{light_bucket}_{context.time_of_day}"
    
    def _update_multi_objective_learning(self, arm: float, metrics: PerformanceMetrics) -> None:
        """Learn from multi-objective performance metrics"""
        # Composite score from multiple objectives
        composite_score = (
            0.3 * metrics.speed_efficiency +
            0.4 * metrics.safety_score +
            0.1 * metrics.exploration_score +
            0.1 * metrics.energy_efficiency +
            0.1 * metrics.mission_progress
        )
        
        # Update value with composite score influence
        self.value[arm] = 0.8 * self.value[arm] + 0.2 * composite_score
    
    def _update_exploration_rate(self) -> None:
        """Dynamically adjust exploration rate - rate limited"""
        current_time = time.time()
        
        # Rate limit exploration adjustments
        if current_time - self.last_log_time < self.log_interval:
            return
        
        # Decay exploration over time
        self.current_eps = max(self.cfg.min_eps, self.current_eps * self.cfg.decay_rate)
        
        # Increase exploration if performance is declining (less aggressive)
        if len(self.performance_history) > 10:
            recent_performance = [m.safety_score for m in list(self.performance_history)[-10:]]
            if recent_performance:
                avg_recent = sum(recent_performance) / len(recent_performance)
                if avg_recent < 0.5:  # Poor recent performance (stricter threshold)
                    # Smaller, capped exploration increase
                    old_eps = self.current_eps
                    self.current_eps = min(self.max_exploration, self.current_eps + self.exploration_step)
                    
                    # Only log if exploration actually changed
                    if self.current_eps > old_eps:
                        self.log.info(f"Increased exploration to {self.current_eps:.3f} due to poor performance")
                        self.last_log_time = current_time
    
    def _check_adaptation_triggers(self) -> None:
        """Check if we need to adapt strategy - rate limited"""
        current_time = time.time()
        
        # Rate limit adaptations
        if current_time - self.last_adaptation_time < self.min_adaptation_interval:
            return
        
        if self.total_actions % 100 == 0 and self.total_actions > 100:
            success_rate = self.successful_actions / self.total_actions
            
            if success_rate < 0.3:  # Stricter threshold for low success rate
                self._trigger_adaptation("low_success_rate")
                self.last_adaptation_time = current_time
            elif len(self.performance_history) > 20:
                recent_safety = [m.safety_score for m in list(self.performance_history)[-20:]]
                if recent_safety and sum(recent_safety) / len(recent_safety) < 0.4:  # Stricter threshold
                    self._trigger_adaptation("safety_concerns")
                    self.last_adaptation_time = current_time
    
    def _trigger_adaptation(self, reason: str) -> None:
        """Trigger learning adaptation"""
        self.adaptation_triggers += 1
        self.log.info(f"Triggering adaptation #{self.adaptation_triggers}: {reason}")
        
        # Reset some learning to allow re-exploration
        for arm in self.cfg.arms:
            if self.count[arm] > 0:
                self.value[arm] *= 0.8  # Reduce confidence in all arms
                self.count[arm] = max(1, self.count[arm] // 2)  # Reset counts partially
        
        # Increase exploration temporarily
        self.current_eps = min(0.4, self.current_eps * 2.0)
        self.learning_rate = min(0.3, self.learning_rate * 1.5)
        
        self.log.info(f"Adaptation complete: eps={self.current_eps:.3f}, lr={self.learning_rate:.3f}")
    
    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics"""
        session_time = time.time() - self.session_start
        return {
            "total_actions": self.total_actions,
            "success_rate": self.successful_actions / max(1, self.total_actions),
            "current_exploration": self.current_eps,
            "learning_rate": self.learning_rate,
            "adaptation_triggers": self.adaptation_triggers,
            "session_time_minutes": session_time / 60,
            "arm_values": dict(self.value),
            "arm_counts": dict(self.count),
            "arm_confidence": dict(self.confidence),
            "best_arm": max(self.cfg.arms, key=lambda k: self.value[k]) if any(self.value.values()) else None
        }
    
    def _save_state(self) -> None:
        """Save learning state to disk"""
        try:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            state = {
                "value": self.value,
                "count": self.count,
                "confidence": self.confidence,
                "current_eps": self.current_eps,
                "learning_rate": self.learning_rate,
                "total_actions": self.total_actions,
                "successful_actions": self.successful_actions,
                "adaptation_triggers": self.adaptation_triggers,
                "session_start": self.session_start
            }
            with open(self.save_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.log.warning(f"Failed to save learning state: {e}")
    
    def _load_state(self) -> None:
        """Load learning state from disk"""
        try:
            if os.path.exists(self.save_path):
                with open(self.save_path, 'r') as f:
                    state = json.load(f)
                
                # Restore state with validation
                if "value" in state:
                    for arm in self.cfg.arms:
                        if str(arm) in state["value"]:
                            self.value[arm] = state["value"][str(arm)]
                        if str(arm) in state.get("count", {}):
                            self.count[arm] = state["count"][str(arm)]
                        if str(arm) in state.get("confidence", {}):
                            self.confidence[arm] = state["confidence"][str(arm)]
                
                self.current_eps = state.get("current_eps", self.cfg.eps)
                self.learning_rate = state.get("learning_rate", 0.1)
                self.total_actions = state.get("total_actions", 0)
                self.successful_actions = state.get("successful_actions", 0)
                self.adaptation_triggers = state.get("adaptation_triggers", 0)
                
                self.log.info(f"Loaded learning state: {self.total_actions} actions, {self.adaptation_triggers} adaptations")
        except Exception as e:
            self.log.warning(f"Failed to load learning state: {e}")


# Keep the original EpsilonGreedyBandit for backward compatibility
class EpsilonGreedyBandit:
    def __init__(self, cfg: BanditConfig) -> None:
        self.cfg = cfg
        self.log = init_logger("picarx.learning")
        self.value: Dict[float, float] = {a: 0.0 for a in cfg.arms}
        self.count: Dict[float, int] = {a: 0 for a in cfg.arms}

    def select(self) -> float:
        if random.random() < self.cfg.eps:
            a = random.choice(self.cfg.arms)
            self.log.debug(f"explore arm={a}")
            return a
        # exploit
        a = max(self.cfg.arms, key=lambda k: self.value[k])
        return a

    def update(self, arm: float, reward: float) -> None:
        n = self.count[arm] + 1
        v = self.value[arm]
        self.count[arm] = n
        self.value[arm] = v + (reward - v) / n
