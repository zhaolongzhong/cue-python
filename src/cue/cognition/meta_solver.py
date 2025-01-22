"""
Meta-cognitive problem-solving system with dynamic strategy adaptation.
"""

import os
import json
import logging
from enum import Enum
from typing import Dict, List
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

class SolutionStrategy(Enum):
    DECOMPOSITION = "decomposition"  # Break into sub-problems
    PATTERN_MATCHING = "pattern_matching"  # Apply known patterns
    EXPLORATION = "exploration"  # Try novel approaches
    OPTIMIZATION = "optimization"  # Refine existing solutions
    INTEGRATION = "integration"  # Combine multiple approaches

@dataclass
class ProblemContext:
    """Represents problem solving context."""
    complexity: float
    constraints: List[str]
    domain: str
    priority: int
    resources: Dict[str, float]

@dataclass
class SolutionMetrics:
    """Metrics for solution evaluation."""
    effectiveness: float
    efficiency: float
    novelty: float
    robustness: float
    adaptability: float

class MetaCognitiveSolver:
    def __init__(self):
        """Initialize meta-cognitive problem solver."""
        self.solution_history: Dict[str, List[Dict]] = {}
        self.strategy_effectiveness: Dict[str, Dict[SolutionStrategy, float]] = {}
        self.learning_patterns: Dict[str, Dict] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load persistent solver state."""
        data_dir = Path(os.path.expanduser("~/.cache/meta_solver"))
        data_file = data_dir / "solver_state.json"

        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            return

        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)
                    self.solution_history = data.get('solution_history', {})

                    # Restore strategy effectiveness
                    self.strategy_effectiveness = {
                        domain: {
                            SolutionStrategy(s): v
                            for s, v in strategies.items()
                        }
                        for domain, strategies in data.get(
                            'strategy_effectiveness', {}).items()
                    }

                    self.learning_patterns = data.get('learning_patterns', {})
            except Exception as e:
                logger.error(f"Error loading solver state: {e}")

    def _save_state(self) -> None:
        """Save solver state persistently."""
        data_dir = Path(os.path.expanduser("~/.cache/meta_solver"))
        data_file = data_dir / "solver_state.json"

        try:
            data = {
                'solution_history': self.solution_history,
                'strategy_effectiveness': {
                    domain: {
                        s.value: v
                        for s, v in strategies.items()
                    }
                    for domain, strategies in self.strategy_effectiveness.items()
                },
                'learning_patterns': self.learning_patterns
            }
            with open(data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving solver state: {e}")

    def analyze_problem(self, problem_id: str,
                       context: ProblemContext) -> Dict:
        """
        Analyze problem and determine solution strategy.
        
        Args:
            problem_id: Unique problem identifier
            context: Problem context
            
        Returns:
            Dict containing analysis results
        """
        # Analyze domain patterns
        domain_patterns = self._analyze_domain_patterns(context.domain)

        # Score strategies
        strategy_scores = self._score_strategies(context, domain_patterns)

        # Select optimal strategy mix
        strategy_mix = self._determine_strategy_mix(
            strategy_scores, context.complexity)

        # Generate solution plan
        solution_plan = self._generate_solution_plan(strategy_mix, context)

        # Record analysis
        if problem_id not in self.solution_history:
            self.solution_history[problem_id] = []

        self.solution_history[problem_id].append({
            'timestamp': datetime.now().isoformat(),
            'context': asdict(context),
            'strategy_mix': {
                s.value: w for s, w in strategy_mix.items()
            },
            'solution_plan': solution_plan
        })

        self._save_state()

        return {
            'strategy_mix': {
                s.value: w for s, w in strategy_mix.items()
            },
            'solution_plan': solution_plan,
            'domain_insights': domain_patterns
        }

    def _analyze_domain_patterns(self, domain: str) -> Dict:
        """Analyze patterns in problem domain."""
        if domain not in self.learning_patterns:
            return {'status': 'new_domain', 'patterns': []}

        patterns = self.learning_patterns[domain]

        # Analyze success patterns
        success_patterns = [
            p for p in patterns.get('solution_patterns', [])
            if p.get('success_rate', 0) > 0.7
        ]

        # Analyze failure patterns
        failure_patterns = [
            p for p in patterns.get('solution_patterns', [])
            if p.get('success_rate', 0) < 0.3
        ]

        # Calculate domain insights
        return {
            'status': 'analyzed',
            'success_patterns': success_patterns,
            'failure_patterns': failure_patterns,
            'domain_maturity': len(patterns.get('solution_patterns', [])) / 100
        }

    def _score_strategies(self, context: ProblemContext,
                         domain_patterns: Dict) -> Dict[SolutionStrategy, float]:
        """Score potential solution strategies."""
        base_scores = {
            SolutionStrategy.DECOMPOSITION: 0.5,
            SolutionStrategy.PATTERN_MATCHING: 0.5,
            SolutionStrategy.EXPLORATION: 0.5,
            SolutionStrategy.OPTIMIZATION: 0.5,
            SolutionStrategy.INTEGRATION: 0.5
        }

        # Adjust for complexity
        if context.complexity > 0.7:
            base_scores[SolutionStrategy.DECOMPOSITION] += 0.2
            base_scores[SolutionStrategy.INTEGRATION] += 0.2
        else:
            base_scores[SolutionStrategy.PATTERN_MATCHING] += 0.2

        # Adjust for domain maturity
        domain_maturity = domain_patterns.get('domain_maturity', 0)
        if domain_maturity > 0.7:
            base_scores[SolutionStrategy.OPTIMIZATION] += 0.2
            base_scores[SolutionStrategy.PATTERN_MATCHING] += 0.1
        else:
            base_scores[SolutionStrategy.EXPLORATION] += 0.2

        # Adjust for resource constraints
        if sum(context.resources.values()) < len(context.resources) * 0.5:
            base_scores[SolutionStrategy.OPTIMIZATION] += 0.2
            base_scores[SolutionStrategy.PATTERN_MATCHING] += 0.1

        # Normalize scores
        total = sum(base_scores.values())
        return {
            strategy: score / total
            for strategy, score in base_scores.items()
        }

    def _determine_strategy_mix(self,
                              scores: Dict[SolutionStrategy, float],
                              complexity: float) -> Dict[SolutionStrategy, float]:
        """Determine optimal mix of strategies."""
        # Start with normalized scores
        strategy_mix = scores.copy()

        # Adjust for complexity
        if complexity > 0.8:
            # Complex problems need more balanced approach
            mean_score = sum(strategy_mix.values()) / len(strategy_mix)
            strategy_mix = {
                s: (v * 0.7 + mean_score * 0.3)
                for s, v in strategy_mix.items()
            }
        elif complexity < 0.3:
            # Simple problems can focus on top strategies
            top_strategy = max(strategy_mix.items(), key=lambda x: x[1])[0]
            strategy_mix = {
                s: (v * 1.3 if s == top_strategy else v * 0.7)
                for s, v in strategy_mix.items()
            }

        # Normalize final mix
        total = sum(strategy_mix.values())
        return {
            strategy: weight / total
            for strategy, weight in strategy_mix.items()
        }

    def _generate_solution_plan(self,
                              strategy_mix: Dict[SolutionStrategy, float],
                              context: ProblemContext) -> List[Dict]:
        """Generate structured solution plan."""
        solution_plan = []

        # Sort strategies by weight
        sorted_strategies = sorted(
            strategy_mix.items(), key=lambda x: x[1], reverse=True)

        for strategy, weight in sorted_strategies:
            if weight < 0.1:  # Skip low-weight strategies
                continue

            if strategy == SolutionStrategy.DECOMPOSITION:
                solution_plan.append({
                    'phase': 'decomposition',
                    'weight': weight,
                    'actions': [
                        'Identify independent components',
                        'Map dependencies',
                        'Create sub-problems'
                    ]
                })

            elif strategy == SolutionStrategy.PATTERN_MATCHING:
                solution_plan.append({
                    'phase': 'pattern_matching',
                    'weight': weight,
                    'actions': [
                        'Search known patterns',
                        'Evaluate pattern applicability',
                        'Adapt patterns to context'
                    ]
                })

            elif strategy == SolutionStrategy.EXPLORATION:
                solution_plan.append({
                    'phase': 'exploration',
                    'weight': weight,
                    'actions': [
                        'Generate novel approaches',
                        'Test hypothetical solutions',
                        'Document new patterns'
                    ]
                })

            elif strategy == SolutionStrategy.OPTIMIZATION:
                solution_plan.append({
                    'phase': 'optimization',
                    'weight': weight,
                    'actions': [
                        'Identify improvement areas',
                        'Optimize resource usage',
                        'Refine solution components'
                    ]
                })

            elif strategy == SolutionStrategy.INTEGRATION:
                solution_plan.append({
                    'phase': 'integration',
                    'weight': weight,
                    'actions': [
                        'Combine solution components',
                        'Verify integrations',
                        'Optimize interfaces'
                    ]
                })

        return solution_plan

    def update_solution_results(self, problem_id: str,
                              metrics: SolutionMetrics) -> Dict:
        """
        Update solution results and learning patterns.
        
        Args:
            problem_id: Problem identifier
            metrics: Solution metrics
            
        Returns:
            Dict containing update results
        """
        if problem_id not in self.solution_history:
            return {
                'status': 'error',
                'message': 'No solution history found'
            }

        # Get latest solution attempt
        latest_solution = self.solution_history[problem_id][-1]
        domain = latest_solution['context']['domain']

        # Update strategy effectiveness
        if domain not in self.strategy_effectiveness:
            self.strategy_effectiveness[domain] = {
                strategy: 0.5 for strategy in SolutionStrategy
            }

        strategy_mix = {
            SolutionStrategy(s): w
            for s, w in latest_solution['strategy_mix'].items()
        }

        for strategy, weight in strategy_mix.items():
            current_effectiveness = self.strategy_effectiveness[domain][strategy]
            # Update with weighted impact
            self.strategy_effectiveness[domain][strategy] = (
                current_effectiveness * 0.7 +
                metrics.effectiveness * weight * 0.3
            )

        # Update learning patterns
        if domain not in self.learning_patterns:
            self.learning_patterns[domain] = {
                'solution_patterns': []
            }

        # Record solution pattern
        self.learning_patterns[domain]['solution_patterns'].append({
            'timestamp': datetime.now().isoformat(),
            'strategy_mix': latest_solution['strategy_mix'],
            'metrics': asdict(metrics),
            'success_rate': metrics.effectiveness
        })

        # Keep last 100 patterns
        if len(self.learning_patterns[domain]['solution_patterns']) > 100:
            self.learning_patterns[domain]['solution_patterns'] = (
                self.learning_patterns[domain]['solution_patterns'][-100:]
            )

        self._save_state()

        return {
            'status': 'updated',
            'domain': domain,
            'effectiveness': metrics.effectiveness,
            'learning_progress': len(
                self.learning_patterns[domain]['solution_patterns']
            ) / 100
        }

    def get_solver_insights(self) -> Dict:
        """
        Get insights about problem-solving effectiveness.
        
        Returns:
            Dict containing solver insights
        """
        # Analyze domain effectiveness
        domain_effectiveness = {}
        for domain, strategies in self.strategy_effectiveness.items():
            avg_effectiveness = sum(strategies.values()) / len(strategies)
            most_effective = max(strategies.items(), key=lambda x: x[1])
            domain_effectiveness[domain] = {
                'average_effectiveness': avg_effectiveness,
                'best_strategy': most_effective[0].value,
                'best_score': most_effective[1]
            }

        # Analyze learning progress
        learning_progress = {}
        for domain, patterns in self.learning_patterns.items():
            if patterns['solution_patterns']:
                recent_patterns = patterns['solution_patterns'][-10:]
                success_rate = sum(
                    p['success_rate'] for p in recent_patterns
                ) / len(recent_patterns)
                learning_progress[domain] = {
                    'patterns_recorded': len(patterns['solution_patterns']),
                    'recent_success_rate': success_rate
                }

        # Get global statistics
        total_solutions = sum(
            len(history) for history in self.solution_history.values()
        )

        return {
            'total_solutions': total_solutions,
            'domains_analyzed': len(self.strategy_effectiveness),
            'domain_effectiveness': domain_effectiveness,
            'learning_progress': learning_progress
        }
