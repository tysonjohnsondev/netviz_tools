# Gallery

Three notebooks, each self-contained and run offline on the bundled sample (`faostat.load_sample`). They are committed with their outputs, so the figures below are the ones the code produced. Use the download link at the top of each notebook page to get the `.ipynb` file.

- [The 2022 wheat shock](wheat-2022-shock.ipynb): Russian and Ukrainian wheat exports in 2021, 2022 and 2023, why the reporting perspective changes the answer, supplier concentration from 2010 to 2024, and a Sankey diagram, flow map and network for 2022.
- [Community structure](community-structure.ipynb): Louvain communities in the 2022 wheat network, their modularity and members, the community graph, stability across random seeds, and a comparison with greedy modularity.
- [Power law vs lognormal](power-law-vs-lognormal.ipynb): tail fits for degree and out-strength in the 2022 wheat, maize and soya bean networks, and what the likelihood-ratio tests do and do not show.

To run them yourself:

```bash
pip install netviz-tools jupyter
jupyter lab docs/gallery/
```
