<?php

class EnjoyNaiveBayes
{
    private array $classes = [];
    private array $theta = [];   // разреженные счётчики токенов по классам
    private array $priors = [];  // априорные вероятности
    private array $classCounts = []; // кол-во документов по классам

    // обучаем новую партию данных
    public function partialFit(array $samples, array $labels)
    {
        foreach ($labels as $i => $c) {
            $sample = $samples[$i];

            if (!isset($this->classes[$c])) {
                $this->classes[$c] = $c;
                $this->theta[$c] = [];      // пустой массив токенов
                $this->priors[$c] = 0.0;
                $this->classCounts[$c] = 0;
            }

            // увеличиваем счётчики токенов
            foreach ($sample as $tokenIndex => $count) {
                $this->theta[$c][$tokenIndex] = ($this->theta[$c][$tokenIndex] ?? 0) + $count;
            }

            $this->classCounts[$c]++;
        }

        // пересчитываем априоры
        $totalDocs = array_sum($this->classCounts);
        foreach ($this->classes as $c) {
            $this->priors[$c] = $this->classCounts[$c] / $totalDocs;
        }
    }

    // предсказание top-N классов
    public function predictTopN(array $samples, int $N = 3): array
    {
        $results = [];

        foreach ($samples as $sample) {
            $scores = [];

            foreach ($this->classes as $c) {
                $log_prior = log($this->priors[$c] ?? 1e-12);
                $log_likelihood = 0.0;

                $thetaC = $this->theta[$c];
                $totalCount = array_sum($thetaC) + count($thetaC); // Laplace

                foreach ($sample as $tokenIndex => $count) {
                    $tokenProb = ($thetaC[$tokenIndex] ?? 0) + 1;
                    $log_likelihood += $count * log($tokenProb / $totalCount);
                }

                $scores[$c] = $log_prior + $log_likelihood;
            }

            // softmax
            $maxLog = max($scores);
            $expScores = array_map(fn($v) => exp($v - $maxLog), $scores);
            $sumExp = array_sum($expScores);
            $probs = array_map(fn($v) => $v / $sumExp, $expScores);

            arsort($probs);
            $results[] = array_slice($probs, 0, $N, true);
        }

        return $results;
    }
}
