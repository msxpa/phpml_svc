<?php

class CustomNaiveBayes
{
    private array $classes = [];
    private array $theta = [];  // вероятность признаков по классам
    private array $priors = []; // априорные вероятности

    public function fit(array $samples, array $labels)
    {
        $this->classes = array_values(array_unique($labels));
        $n_classes = count($this->classes);
        $n_features = count($samples[0]);

        $this->theta = [];
        $this->priors = [];

        foreach ($this->classes as $c) {
            // индексы документов с классом $c
            $indices = array_keys($labels, $c);

            // выбираем все примеры этого класса
            $class_samples = array_map(fn($i) => $samples[$i], $indices);

            // суммируем признаки и добавляем Laplace smoothing
            $feature_counts = [];
            for ($j = 0; $j < $n_features; $j++) {
                $sum = array_sum(array_column($class_samples, $j)) + 1; // +1 Laplace
                $feature_counts[] = $sum;
            }

            $total_count = array_sum($feature_counts);

            // вычисляем вероятность признака
            $this->theta[$c] = array_map(fn($v) => $v / $total_count, $feature_counts);

            // априорная вероятность
            $this->priors[$c] = count($indices) / count($labels);
        }
    }

    public function partialFit(array $samples, array $labels)
    {
        if (empty($this->classes)) {
            // первая инициализация
            $this->fit($samples, $labels);
            return;
        }

        $n_features = count($samples[0]);

        foreach (array_unique($labels) as $c) {
            if (!in_array($c, $this->classes)) {
                $this->classes[] = $c;
                $this->theta[$c] = array_fill(0, $n_features, 1e-6); // малые значения для новых признаков
                $this->priors[$c] = 0;
            }
        }

        // обновляем feature counts и priors
        $classCounts = array_count_values($labels);
        foreach ($samples as $i => $sample) {
            $c = $labels[$i];
            foreach ($sample as $j => $count) {
                $this->theta[$c][$j] += $count;
            }
        }

        // обновляем априоры
        $totalSamples = array_sum(array_map('array_sum', [$classCounts]));
        foreach ($this->classes as $c) {
            $this->priors[$c] = ($this->priors[$c] * ($totalSamples - count($labels)) + ($classCounts[$c] ?? 0)) / $totalSamples;
        }

        // нормализуем theta
        foreach ($this->classes as $c) {
            $sumTheta = array_sum($this->theta[$c]);
            $this->theta[$c] = array_map(fn($v) => $v / $sumTheta, $this->theta[$c]);
        }
    }


    public function predictTopN(array $samples, int $N = 3): array
    {
        $results = [];

        foreach ($samples as $sample) {
            $scores = [];

            foreach ($this->classes as $c) {
                $log_prior = log($this->priors[$c]);
                $log_likelihood = 0;

                foreach ($sample as $i => $count) {
                    if ($count > 0) {
                        $log_likelihood += $count * log($this->theta[$c][$i]);
                    }
                }

                $scores[$c] = $log_prior + $log_likelihood;
            }

            // softmax
            $maxLog = max($scores);
            $expScores = array_map(fn($v) => exp($v - $maxLog), $scores);
            $sumExp = array_sum($expScores);
            $probs = array_map(fn($v) => $v / $sumExp, $expScores);

            // сортировка top-N
            arsort($probs);
            $results[] = array_slice($probs, 0, $N, true);
        }

        return $results;
    }
}
