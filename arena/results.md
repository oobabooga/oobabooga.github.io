# Preset Arena results

Description of the experiment: [Preset Arena: 17,205 comparisons between 241 different presets.](https://www.reddit.com/r/LocalLLaMA/comments/14adfw2/preset_arena_17205_comparisons_between_241/)

Some numbers:

* 7215 valid votes
* 951 voting sessions
* 288 users with usernames

## Final results

* Sorted by instruct performance: [instruct](https://oobabooga.github.io/arena/instruct.html)
* Sorted by chat performance: [chat](https://oobabooga.github.io/arena/chat.html)

Preset definitions: [presets](https://oobabooga.github.io/arena/presets.html)

## Dataset

https://huggingface.co/datasets/oobabooga/preset-arena

## Ranking the presets

### Bad voting sessions

The first step in the analysis of the votes was to try to identify suspicious voters. Each voting session received a unique uuid string, allowing the frequency of left/right votes to be analyzed.

I have used the following code to calculate the probability that a voting session was biased. It was obtained by asking ChatGPT for a fair coin test:

```python
from scipy.stats import beta

def compute_bias_probability(outcomes, prior_alpha=1, prior_beta=1, _print=False):
    # Count the number of heads and tails
    num_heads = outcomes.count('left')
    num_tails = outcomes.count('right')

    if _print:
        print(num_heads, num_tails)

    # Update the prior with the observed outcomes
    posterior_alpha = prior_alpha + num_heads
    posterior_beta = prior_beta + num_tails

    # Calculate the bias probability using the Beta distribution
    bias_probability = beta.cdf(0.5, posterior_alpha, posterior_beta)

    return bias_probability
```

A session was disconsidered if `bias_probability > 0.99`, which happened for 0.6% of all sessions.

### Estimating the elo scores

The basic formula is

```python
def update_rating(rating, opponent_rating, outcome, k=32):
    expected_score = 1 / (1 + 10**((opponent_rating - rating) / 400))
    new_rating = rating + k * (outcome - expected_score)
    return new_rating
```

where the ratings are initialized as `1000` for all presets, and `outcome` is 1 in case of winning and 0 in case of losing.

To make things more robust, I have used the following procedure instead of calculating the elo scores just once:

* take a random subsample containing 90% of votes
* using that sample, calculate the elo scores for chat and instruct prompts separately
* repeat 200 times
* take the averages of the elo scores for each preset

Additionally, I have not counted votes where both completions are identical.

### Comments

1) I find that the top chat presets are all kind of the same. It may be due to the chat prompts being too simple and short, causing presets with low top_p to be favored.

2) 5 variations of the Mirostat preset were included. It turned out that `Mirostat-5` was a bit better than the `Mirostat` preset originally included in text-generation-webui:

<table><tr><th>preset</th><th>params</th><th>elo score (chat)</th><th>elo score (instruct)</th><th>elo score (all)</th><th>matches (chat)</th><th>matches (instruct)</th></tr><tr><td>Mirostat-5</td><td>2</td><td>1012.723756636154</td><td>1100.0171006055577</td><td>1056.3704286208558</td><td>36</td><td>23</td></tr><tr><td>Mirostat</td><td>1</td><td>993.0564327577029</td><td>1109.172602933306</td><td>1051.1145178455045</td><td>27</td><td>22</td></tr><tr><td>Mirostat-2</td><td>2</td><td>1067.8824770156248</td><td>1028.214156025321</td><td>1048.0483165204728</td><td>29</td><td>25</td></tr><tr><td>Mirostat-4</td><td>2</td><td>1031.9219927236945</td><td>1020.1965461643792</td><td>1026.059269444037</td><td>37</td><td>35</td></tr><tr><td>Mirostat-3</td><td>2</td><td>988.1664164954003</td><td>1021.2103791101517</td><td>1004.6883978027761</td><td>29</td><td>29</td></tr></table>

3) Similarly, 5 Contrastive Search variations were included, `Contrastive Search-3` ended up being a bit better than the original `Contrastive Search`:

<table><tr><th>preset</th><th>params</th><th>elo score (chat)</th><th>elo score (instruct)</th><th>elo score (all)</th><th>matches (chat)</th><th>matches (instruct)</th></tr><tr><td>Special-Contrastive Search-3</td><td>3</td><td>1077.6702759297164</td><td>1115.8151721393688</td><td>1096.7427240345426</td><td>27</td><td>18</td></tr><tr><td>Special-Contrastive Search</td><td>3</td><td>1077.3415040295642</td><td>1095.4654729538931</td><td>1086.4034884917287</td><td>35</td><td>31</td></tr><tr><td>Special-Contrastive Search-1</td><td>3</td><td>899.7205727080627</td><td>851.8635177853589</td><td>875.7920452467108</td><td>16</td><td>10</td></tr><tr><td>Special-Contrastive Search-4</td><td>3</td><td>765.788679774467</td><td>790.9640810990088</td><td>778.3763804367379</td><td>33</td><td>19</td></tr><tr><td>Special-Contrastive Search-2</td><td>3</td><td>801.0156035678388</td><td>736.8621355164904</td><td>768.9388695421646</td><td>27</td><td>25</td></tr></table>

4) Eta Sampling (another special technique) did not perform very well (~but its parameters are present in other top-performing presets~):

<table><tr><th>preset</th><th>params</th><th>elo score (chat)</th><th>elo score (instruct)</th><th>elo score (all)</th><th>matches (chat)</th><th>matches (instruct)</th></tr><tr><td>Special-Eta Sampling</td><td>3</td><td>1018.5269796896921</td><td>1016.4519009597249</td><td>1017.4894403247085</td><td>29</td><td>25</td></tr></table>

5) The best preset overall, considering the average of the chat and instruct elo scores, was also perhaps the most obvious. I originally named it `simple-1` not expecting it to get anywhere:

```
temperature: 0.7
top_p: 0.9
repetition_penalty: 1.15
top_k: 20
```

The StarChat preset, also very simple, also performed well:

```
temperature: 0.2
top_p: 0.95
top_k: 50
```

This demonstrates that fancy samplers may not be that necessary.

### Presets that I chose

For the purpose of including better presets in text-generation-webui, I removed presets with `top_p < 0.05` or `top_k < 3` because that seemed too low and artificial. That left me with the following (in decreasing order of elo score):

#### Instruct

| Preset | New name |
|------|---------|
| random_preset_066 | Divine Intellect |
| random_preset_134 | Big O |
| simple-1 | |
| random_preset_035 | Space Alien |
| starchat | StarChat |
| random_preset_183 | Titanic |
| tfs-with-top-a | | 
| random_preset_002 | Asterism |
| Special-Contrastive Search-3 | Contrastive Search |

#### Chat

| Preset | New name |
|------|---------|
| random_preset_101 | Midnight Enigma |
| random_preset_161 | Yara |
| random_preset_120 | Shortwave |
| Kobold-Godlike | |

I took the liberty of giving gave some cheesy names for the new random presets.

### Sampler frequency (outdated, see below)

~In those 13 new presets, these are the sampling parameters that are present and how many times they appear:~

```
     12 temperature
     11 top_p
     11 top_k
     11 repetition_penalty
      5 top_a
      3 tfs
      2 typical_p
      2 eta_cutoff
      1 penalty_alpha
      1 epsilon_cutoff
      1 encoder_repetition_penalty
```

### Sampler frequency (updated)

In a follow-up analysis, I have tried removing samplers from the presets and seeing if the resulting logits changed. 

For that, I took some random story that I copied and pasted from the internet, split it by spaces, and computed the logits using as input the first N words for N <= 200. That is, 200 logit vectors were computed for each preset. Then I considered a parameter as redundant if its removal kept the logits identical 90% of the time or more.

The resulting parameter frequency after this clean-up was:

```
     12 temperature
     11 top_p
     11 top_k
     11 repetition_penalty
      2 typical_p
      2 tfs
      1 top_a
      1 penalty_alpha
      1 encoder_repetition_penalty
```

Note that the eta sampling parameters (`epsilon_cutoff` and `eta_cutoff`) disappeared completely. 
