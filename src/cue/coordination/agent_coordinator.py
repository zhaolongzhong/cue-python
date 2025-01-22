"""
Advanced multi-agent coordination system with dynamic role management.
"""

import os
import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    COORDINATOR = "coordinator"
    SPECIALIST = "specialist"
    SUPPORT = "support"
    LEARNER = "learner"
    ANALYZER = "analyzer"

@dataclass
class AgentState:
    """Represents agent's current state."""
    role: AgentRole
    capabilities: List[str]
    performance: float
    reliability: float
    availability: bool
    last_active: str

@dataclass
class TaskRequirements:
    """Represents task requirements."""
    complexity: float
    specialization: List[str]
    priority: int
    deadline: Optional[str]

class AgentCoordinator:
    def __init__(self):
        """Initialize agent coordination system."""
        self.agent_states: Dict[str, AgentState] = {}
        self.task_history: Dict[str, List[Dict]] = {}
        self.coordination_metrics: Dict[str, Dict] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load persistent coordination state."""
        data_dir = Path(os.path.expanduser("~/.cache/agent_coordinator"))
        data_file = data_dir / "coordination_state.json"

        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            return

        if data_file.exists():
            try:
                with open(data_file) as f:
                    data = json.load(f)

                    # Restore agent states
                    self.agent_states = {
                        k: AgentState(
                            role=AgentRole(v['role']),
                            capabilities=v['capabilities'],
                            performance=v['performance'],
                            reliability=v['reliability'],
                            availability=v['availability'],
                            last_active=v['last_active']
                        )
                        for k, v in data.get('agent_states', {}).items()
                    }

                    self.task_history = data.get('task_history', {})
                    self.coordination_metrics = data.get('coordination_metrics', {})
            except Exception as e:
                logger.error(f"Error loading coordination state: {e}")

    def _save_state(self) -> None:
        """Save coordination state persistently."""
        data_dir = Path(os.path.expanduser("~/.cache/agent_coordinator"))
        data_file = data_dir / "coordination_state.json"

        try:
            data = {
                'agent_states': {
                    k: asdict(v) for k, v in self.agent_states.items()
                },
                'task_history': self.task_history,
                'coordination_metrics': self.coordination_metrics
            }
            with open(data_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error saving coordination state: {e}")

    def register_agent(self, agent_id: str,
                      capabilities: List[str],
                      initial_role: Optional[AgentRole] = None) -> Dict:
        """
        Register a new agent in the coordination system.
        
        Args:
            agent_id: Unique agent identifier
            capabilities: List of agent capabilities
            initial_role: Optional initial role assignment
            
        Returns:
            Dict containing registration results
        """
        if not initial_role:
            initial_role = self._determine_optimal_role(capabilities)

        state = AgentState(
            role=initial_role,
            capabilities=capabilities,
            performance=0.5,  # Initial neutral performance
            reliability=0.5,  # Initial neutral reliability
            availability=True,
            last_active=datetime.now().isoformat()
        )

        self.agent_states[agent_id] = state

        # Initialize metrics tracking
        if agent_id not in self.coordination_metrics:
            self.coordination_metrics[agent_id] = {
                'tasks_completed': 0,
                'success_rate': 0.0,
                'avg_performance': 0.5,
                'reliability_history': []
            }

        self._save_state()

        return {
            'status': 'registered',
            'role': initial_role.value,
            'capabilities': capabilities
        }

    def _determine_optimal_role(self, capabilities: List[str]) -> AgentRole:
        """Determine optimal role based on capabilities."""
        capability_weights = {
            'coordination': (AgentRole.COORDINATOR, 0.8),
            'analysis': (AgentRole.ANALYZER, 0.7),
            'learning': (AgentRole.LEARNER, 0.6),
            'support': (AgentRole.SUPPORT, 0.5)
        }

        role_scores = {role: 0.0 for role in AgentRole}

        for capability in capabilities:
            for key, (role, weight) in capability_weights.items():
                if key in capability.lower():
                    role_scores[role] += weight

        # Default to specialist if no strong matches
        if max(role_scores.values()) < 0.5:
            return AgentRole.SPECIALIST

        return max(role_scores.items(), key=lambda x: x[1])[0]

    def update_agent_state(self, agent_id: str,
                          metrics: Dict[str, Any]) -> Dict:
        """
        Update agent state based on performance metrics.
        
        Args:
            agent_id: Agent identifier
            metrics: Performance metrics
            
        Returns:
            Dict containing update results
        """
        if agent_id not in self.agent_states:
            return {
                'status': 'error',
                'message': 'Agent not registered'
            }

        current_state = self.agent_states[agent_id]

        # Update performance metrics
        performance = metrics.get('performance', current_state.performance)
        reliability = metrics.get('reliability', current_state.reliability)

        # Check for role optimization
        new_role = self._optimize_role(
            agent_id, performance, reliability, current_state.role)

        # Update state
        self.agent_states[agent_id] = AgentState(
            role=new_role,
            capabilities=current_state.capabilities,
            performance=performance,
            reliability=reliability,
            availability=metrics.get('availability', current_state.availability),
            last_active=datetime.now().isoformat()
        )

        # Update coordination metrics
        self._update_metrics(agent_id, metrics)

        self._save_state()

        return {
            'status': 'updated',
            'new_role': new_role.value,
            'performance': performance,
            'reliability': reliability
        }

    def _optimize_role(self, agent_id: str,
                      performance: float,
                      reliability: float,
                      current_role: AgentRole) -> AgentRole:
        """Optimize agent role based on performance."""
        if performance < 0.3 or reliability < 0.3:
            return AgentRole.SUPPORT  # Move to support role if struggling

        if performance > 0.8 and reliability > 0.8:
            if current_role != AgentRole.COORDINATOR:
                return AgentRole.COORDINATOR  # Promote to coordinator

        if performance > 0.7 and current_role == AgentRole.SUPPORT:
            return AgentRole.SPECIALIST  # Promote from support

        return current_role  # Maintain current role

    def _update_metrics(self, agent_id: str,
                       metrics: Dict[str, Any]) -> None:
        """Update coordination metrics for an agent."""
        if agent_id not in self.coordination_metrics:
            self.coordination_metrics[agent_id] = {
                'tasks_completed': 0,
                'success_rate': 0.0,
                'avg_performance': 0.5,
                'reliability_history': []
            }

        agent_metrics = self.coordination_metrics[agent_id]

        # Update task metrics
        if metrics.get('task_completed', False):
            agent_metrics['tasks_completed'] += 1

        # Update success rate
        success = metrics.get('task_success', False)
        total_tasks = agent_metrics['tasks_completed']
        agent_metrics['success_rate'] = (
            (agent_metrics['success_rate'] * (total_tasks - 1) + int(success))
            / total_tasks if total_tasks > 0 else 0.0
        )

        # Update average performance
        performance = metrics.get('performance', 0.5)
        agent_metrics['avg_performance'] = (
            agent_metrics['avg_performance'] * 0.7 + performance * 0.3
        )

        # Update reliability history
        reliability = metrics.get('reliability', 0.5)
        agent_metrics['reliability_history'].append({
            'timestamp': datetime.now().isoformat(),
            'value': reliability
        })

        # Keep last 100 reliability entries
        if len(agent_metrics['reliability_history']) > 100:
            agent_metrics['reliability_history'] = (
                agent_metrics['reliability_history'][-100:]
            )

    def assign_task(self, task_id: str,
                   requirements: TaskRequirements) -> Dict:
        """
        Assign task to optimal agent based on requirements.
        
        Args:
            task_id: Task identifier
            requirements: Task requirements
            
        Returns:
            Dict containing assignment results
        """
        # Find available agents
        available_agents = {
            agent_id: state
            for agent_id, state in self.agent_states.items()
            if state.availability
        }

        if not available_agents:
            return {
                'status': 'error',
                'message': 'No available agents'
            }

        # Score agents for task
        agent_scores = {}
        for agent_id, state in available_agents.items():
            score = self._calculate_agent_score(state, requirements)
            agent_scores[agent_id] = score

        if not agent_scores:
            return {
                'status': 'error',
                'message': 'No suitable agents found'
            }

        # Select best agent
        selected_agent = max(agent_scores.items(), key=lambda x: x[1])[0]

        # Record assignment
        if task_id not in self.task_history:
            self.task_history[task_id] = []

        self.task_history[task_id].append({
            'timestamp': datetime.now().isoformat(),
            'agent_id': selected_agent,
            'requirements': asdict(requirements),
            'agent_score': agent_scores[selected_agent]
        })

        self._save_state()

        return {
            'status': 'assigned',
            'agent_id': selected_agent,
            'score': agent_scores[selected_agent]
        }

    def _calculate_agent_score(self, state: AgentState,
                             requirements: TaskRequirements) -> float:
        """Calculate agent's suitability score for task."""
        # Base score from performance and reliability
        base_score = (state.performance * 0.6 + state.reliability * 0.4)

        # Capability match score
        capability_match = len(
            set(state.capabilities) & set(requirements.specialization)
        ) / max(len(requirements.specialization), 1)

        # Complexity handling score
        complexity_score = 1.0 - abs(state.performance - requirements.complexity)

        # Role suitability
        role_weights = {
            AgentRole.COORDINATOR: 1.0,
            AgentRole.SPECIALIST: 0.9,
            AgentRole.ANALYZER: 0.8,
            AgentRole.LEARNER: 0.7,
            AgentRole.SUPPORT: 0.6
        }
        role_score = role_weights[state.role]

        # Calculate weighted score
        final_score = (
            base_score * 0.4 +
            capability_match * 0.3 +
            complexity_score * 0.2 +
            role_score * 0.1
        )

        return min(1.0, max(0.0, final_score))

    def get_coordination_summary(self) -> Dict:
        """
        Get summary of coordination system state.
        
        Returns:
            Dict containing coordination summary
        """
        # Calculate role distribution
        role_distribution = {
            role.value: len([
                1 for state in self.agent_states.values()
                if state.role == role
            ])
            for role in AgentRole
        }

        # Calculate system metrics
        total_tasks = sum(
            metrics['tasks_completed']
            for metrics in self.coordination_metrics.values()
        )

        avg_success_rate = (
            sum(metrics['success_rate']
                for metrics in self.coordination_metrics.values()) /
            len(self.coordination_metrics)
            if self.coordination_metrics else 0.0
        )

        avg_performance = (
            sum(metrics['avg_performance']
                for metrics in self.coordination_metrics.values()) /
            len(self.coordination_metrics)
            if self.coordination_metrics else 0.0
        )

        # Get recent task history
        recent_tasks = []
        for task_id, history in self.task_history.items():
            if history:  # Get most recent assignment
                recent_tasks.append({
                    'task_id': task_id,
                    'agent_id': history[-1]['agent_id'],
                    'timestamp': history[-1]['timestamp']
                })

        # Sort by timestamp and get last 10
        recent_tasks.sort(key=lambda x: x['timestamp'], reverse=True)
        recent_tasks = recent_tasks[:10]

        return {
            'active_agents': len(self.agent_states),
            'role_distribution': role_distribution,
            'total_tasks': total_tasks,
            'avg_success_rate': avg_success_rate,
            'avg_performance': avg_performance,
            'recent_tasks': recent_tasks
        }
