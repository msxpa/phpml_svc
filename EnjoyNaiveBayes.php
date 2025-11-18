<?php

class EnjoyNaiveBayes
{
    // Массив классов, например ['electronics' => 'electronics', 'books' => 'books']
    private array $classes = [];

    // Разреженные счётчики токенов по классам:
    // theta['electronics'][123] = 5 → токен 123 встречался 5 раз в документах класса 'electronics'
    private array $theta = [];

    // Априорные вероятности классов P(class)
    // priors['electronics'] = 0.3 → 30% документов относятся к 'electronics'
    private array $priors = [];

    // Количество документов по классам
    // classCounts['electronics'] = 100000 → 100000 документов класса 'electronics' обработано
    private array $classCounts = [];

    // ---- Метод обучения на новой партии данных ----
    public function partialFit(array $samples, array $labels)
    {
        // $samples = [
        //   [12=>1, 345=>2],   // sparse vector документа 1
        //   [5=>3, 123=>1],    // sparse vector документа 2
        // ]
        // $labels = ['electronics', 'books']

        foreach ($labels as $i => $c) { // перебираем документы и их классы
            $sample = $samples[$i];      // sparse vector для документа $i, например [12=>1, 345=>2]

            // Если класс встретился впервые — инициализируем структуры
            if (!isset($this->classes[$c])) {
                $this->classes[$c] = $c;
                $this->theta[$c] = [];      // пустой массив токенов
                $this->priors[$c] = 0.0;    // априор пока 0
                $this->classCounts[$c] = 0; // документов ещё нет
            }

            // Увеличиваем счётчики токенов для этого класса
            foreach ($sample as $tokenIndex => $count) {
                // если токена ещё нет — ставим 0, потом добавляем count
                $this->theta[$c][$tokenIndex] = ($this->theta[$c][$tokenIndex] ?? 0) + $count;
            }

            // Увеличиваем количество документов класса
            $this->classCounts[$c]++;
        }

        // Пересчитываем априорные вероятности для всех классов
        $totalDocs = array_sum($this->classCounts); // общее число документов
        foreach ($this->classes as $c) {
            $this->priors[$c] = $this->classCounts[$c] / $totalDocs;
            // пример: classCounts['electronics']=100000, totalDocs=35000000 → priors['electronics'] ≈ 0.002857
        }
    }

    // ---- Метод предсказания top-N классов для документов ----
    public function predictTopN(array $samples, int $N = 3): array
    {
        $results = [];

        foreach ($samples as $sample) {
            // $sample = [12=>1, 345=>2] — sparse vector документа
            $scores = [];

            foreach ($this->classes as $c) {
                // логарифм априора (чтобы умножение вероятностей стало сложением логов)
                $log_prior = log($this->priors[$c] ?? 1e-12);
                // 1e-12 → защита от деления на ноль, если класс ещё не встречался

                $log_likelihood = 0.0;

                $thetaC = $this->theta[$c]; // все токены класса
                $totalCount = array_sum($thetaC) + count($thetaC); // Laplace smoothing

                foreach ($sample as $tokenIndex => $count) {
                    // Вероятность токена в классе с Laplace
                    $tokenProb = ($thetaC[$tokenIndex] ?? 0) + 1;
                    $log_likelihood += $count * log($tokenProb / $totalCount);
                    // если токен 12 встречался 5 раз, count=1 → log(6/totalCount)
                }

                // Суммируем log_prior + log_likelihood → логарифм полной вероятности класса
                $scores[$c] = $log_prior + $log_likelihood;
            }

            // Преобразуем лог-вероятности в нормированные вероятности через softmax
            $maxLog = max($scores);
            $expScores = array_map(fn($v) => exp($v - $maxLog), $scores);
            $sumExp = array_sum($expScores);
            $probs = array_map(fn($v) => $v / $sumExp, $expScores);

            // сортируем и берём top-N
            arsort($probs);
            $results[] = array_slice($probs, 0, $N, true);
            // пример: ['electronics'=>0.45, 'books'=>0.33, 'clothing'=>0.12]
        }

        return $results;
    }
}
