- https://github.com/dimikout3/MarsExplorer:
  Randomized environments 

- Hilbert curve
  https://en.wikipedia.org/wiki/Hilbert_curve

- Simulator can run 25fps, 11 hours per million frames
- Add computation time to that
- Not too bad, but compare to numbers for generalization:

```
Pratade med Viktor om simulatorn. Han sa att man kan komma upp i ungefär 25fps per instans vilket ger ungefär 11 timmar för en miljon tidssteg *utan träning*. Osäker på hur mycket man måste slänga på för träningen av själva modellen men som ett (väldigt) positivt antagande kan vi räkna med att den tiden är försumbar.

Rikard verkar ha tränat i 12 miljoner tidssteg, vilket motsvarar ungefär 6 dagar.

Experiment på ProcGen (https://arxiv.org/abs/1912.01588) 
```

- Fix a set of training environment instances and test environment instances (by setting initial seed)
- Baselines can use training data as well (for example some nearest neighbor setup)




- Best options so far:

- Gridworld
- UnrealCV
- ViZDoom (zoom is available, might not be exactly what we are looking for)