# Decisões:

## Batch size

- Mais relevante é o desempenho da GPU, e por conta disso precisa ser potência de $2$. Valores considerados: $\text{bs} \in \{ 8, 16, 32, 64, 128 \}.$

## Learning rate

- Valores aleatórios amostrados log-uniforme entre $0.0005$ e $0.1$.
- $50$ candidatos gerados. Cada um treinou por no máximo $2000$ steps (aprox. $20$ épocas com bs 32).