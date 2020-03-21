from dataclasses import dataclass
from dataclasses import field
import functools
import networkx as nx
import random
import rpy2.robjects as ro
from rpy2.robjects.lib import dplyr
from types import SimpleNamespace
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple


class Population:
    """A population with interconnected members.

    This is a state model with each individual being one of:
    - susceptible
    - incubating
    - sick
    - recovered

    If in the jupyter notebook (or compatible environments),
    instances can be displayed as a graph with colored nodes.
    """

    def __init__(self, n_pop:int, m:int = 5, p:float = 1/3):
        """
        Args:
          n_pop: size of the population
          m: parameter m in networkx.powerlaw_cluster_graph()
          p: parameter p in networkx.powerlaw_cluster_graph()
        """
        self.network = nx.powerlaw_cluster_graph(n_pop, m, p)
        self.susceptible = set(self.network.nodes.keys())
        self.incubating: Dict[int, int] = {}
        self.sick: Dict[int, int] = {}
        self.recovered: Set[int] = set()

    def reset(self):
        """Reset the state of the population.

        This will bring back everyone to "susceptible"."""
        self.susceptible = set(self.network.nodes.keys())
        self.incubating = {}
        self.sick = {}
        self.recovered = set()

    def _repr_any_(self, size):
        ag = nx.nx_agraph.to_agraph(self.network)
        ag.node_attr['shape'] = 'point'
        ag.node_attr['color'] = '#b0b0b0b0'
        ag.edge_attr['color'] = '#b0b0b0b0'
        ag.graph_attr['size'] = size
        ag.layout('neato')
        for i in self.incubating:
            n = ag.get_node(i)
            n.attr['color'] = 'yellow'
            n.attr['fillcolor'] = 'orange'
        for i in self.sick:
            n = ag.get_node(i)
            n.attr['color'] = 'orange'
            n.attr['fillcolor'] = 'red'
        for i in self.recovered:
            n = ag.get_node(i)
            n.attr['color'] = 'black'
        return ag

    def _repr_svg_(self):
        ag = self._repr_any_(size='7.75,10.25')
        
        return (ag.draw(format='svg')
                .decode('utf-8'))

    def _repr_png_(self):
        ag = self._repr_any_(size='7.75,10.25')
        return (ag.draw(format='svg')
                .decode('utf-8'))  


class Disease(SimpleNamespace):

    def __init__(self,
                 contagiousness: float,
                 duration_incubation: Callable[[], float] = functools.partial(
                     random.lognormvariate, 1.2, 0.5
                 ),
                 duration_sickness: Callable[[], float] = functools.partial(
                     random.lognormvariate, 1.2, 0.5
                 )):
        self.contagiousness = contagiousness
        self.duration_incubation = duration_incubation
        self.duration_sickness = duration_sickness


@dataclass
class Monitor:
    day: List[int] = field(default_factory=list)
    susceptible: List[int] = field(default_factory=list)
    incubating: List[int] = field(default_factory=list)
    sick: List[int] = field(default_factory=list)

    def record(self, day, population):
        self.day.append(day)
        self.susceptible.append(len(population.susceptible))
        self.incubating.append(len(population.incubating))
        self.sick.append(len(population.sick))


def update_incubations(population: Population,
                       disease: Disease) -> Tuple[
                           List[int], List[int]]:
    """Update incubation status.

    - Do currently-incubating patients contaminate others in their network ?
    - Asymptomatic contamination progress (until symptomatic disease).

    Returns:
      A tuple with a list of new contaminations and a list of new sicknesses."""
    new_contaminations: List[int] = []
    new_sicknesses: List[int] = []
    for case, days_to_sickness in population.incubating.items():
        for person in population.network.neighbors(case):
            if person not in population.susceptible:
                continue
            if random.random() <= disease.contagiousness:
                population.susceptible.remove(person)
                new_contaminations.append(person)
        if days_to_sickness == 0:
            new_sicknesses.append(case)
        else:
            population.incubating[case] = days_to_sickness - 1
    return (new_contaminations, new_sicknesses)


def update_sicknesses(population: Population) -> List[int]:
    """Update sicknesses.

    Symptomatic diseases progress (toward recovery).
    """
    new_recoveries = []
    for case, days_to_recovery in population.sick.items():
        if days_to_recovery == 0:
            new_recoveries.append(case)
        else:
            population.sick[case] -= 1
    return new_recoveries


def update_population(population: Population,
                      disease: Disease,
                      new_contaminations: List[int],
                      new_sicknesses: List[int],
                      new_recoveries: List[int],) -> None:
    """Update a population with the state of new contaminations, new
    sickenesses, and new recoveries."""
    for case in new_sicknesses:
        del population.incubating[case]
        population.sick[case] = round(disease.duration_incubation())
    for case in new_contaminations:
        population.incubating[case] = round(disease.duration_sickness())
    for case in new_recoveries:
        del population.sick[case]
        population.recovered.add(case)


def build_dataframe(monitor: Monitor) -> dplyr.DataFrame:
    what = (
        'susceptible',
        'incubating',
        'sick',
    )
    dataf = dplyr.DataFrame({
        'what': ro.StrVector([v for v in what for x in monitor.day]),
        'day': ro.IntVector([v for x in what for v in monitor.day]),
        'count': ro.IntVector([v for x in what for v in getattr(monitor, x)])
    })
    return dataf


def connections_to_cancel(population: Population,
                          max_size: int,
                          min_connections: int = 5) -> List[Tuple[int, int]]:
    cancelled_connections = []
    for person, n_connections in iter(population.network.degree):
        if n_connections >= max_size:
            for i, neighbor in enumerate(population.network.neighbors(person)):
                if (n_connections - i) <= min_connections:
                    break
                cancelled_connections.append((person, neighbor))
    return cancelled_connections


def simulate_cancelled_events(population: Population,
                              disease: Disease,
                              max_size: int,
                              simulate_day: Callable[[Population, Disease], None],
                              initial_cases: Dict[int, int],
                              delay:int = 0,
                              ndays:int = 3*30) -> Monitor:
    population.reset()
    for case in initial_cases:
        population.incubating[case] = initial_cases[case]

    monitor = Monitor()

    for day in range(delay):
        monitor.record(day, population)
        simulate_day(population, disease)

    cancelled_connections = connections_to_cancel(population, max_size)
    population.network.remove_edges_from(cancelled_connections)

    for day in range(delay, ndays):
        monitor.record(day, population)
        simulate_day(population, disease)

    # Restore the connections for cancelled_events.
    population.network.add_edges_from(cancelled_connections)
    return monitor
