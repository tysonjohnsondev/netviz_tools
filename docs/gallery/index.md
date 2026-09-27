# Gallery

Five notebooks, each self-contained and run offline on the sample bundled with the package. They are committed with their outputs, so the figures below are the ones the code produced. Use the download link at the top of each notebook page to get the `.ipynb` file.

- [Visualizing any NetworkX graph](any-networkx-graph.ipynb): the plots on graphs you already have (karate club, Les Misérables, Florentine families, directed, multi and bipartite graphs, your own positions, a 3,000-node graph), `plot.auto`, and a graph animated over time.
- [From raw FAOSTAT rows to charts](faostat-pipeline.ipynb): the whole pipeline on published FAOSTAT rows with a few added problems: what `clean` fixes and reports, mirror flows, then the largest drops in wheat exports, one country's exports and imports, and soya bean flows animated on a map from 2010 to 2024.
- [The 2022 wheat shock](wheat-2022-shock.ipynb): Russian and Ukrainian wheat exports in 2021, 2022 and 2023, why the reporting perspective changes the answer, which exporters rose and fell, supplier concentration from 2010 to 2024, and a Sankey diagram, flow map and networks.
- [Community structure](community-structure.ipynb): Louvain communities in the 2022 wheat network, their modularity and members, the community graph and adjacency matrix, stability across random seeds, and a comparison with greedy modularity.
- [Power law vs lognormal](power-law-vs-lognormal.ipynb): tail fits for degree and out-strength in the 2022 wheat, maize and soya bean networks, and what the likelihood-ratio tests do and do not show.

To run them yourself:

```bash
pip install netviz-tools jupyter
jupyter lab docs/gallery/
```
