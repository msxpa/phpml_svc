#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';

use Phpml\ModelManager;
use Phpml\Pipeline;
use Phpml\Tokenization\Tokenizer;

// === Кастомный токенизатор ===
class NGramTokenizer implements Tokenizer {
    private int $n;
    public function __construct(int $n = 2) { $this->n = $n; }
    public function tokenize(string $text): array {
        $words = preg_split('/\s+/u', mb_strtolower($text));
        $ngrams = [];
        $count = count($words);
        for ($i = 0; $i < $count; $i++) {
            $ngrams[] = $words[$i];
            if ($i < $count - 1 && $this->n >= 2) {
                $ngrams[] = $words[$i] . ' ' . $words[$i + 1];
            }
        }
        return $ngrams;
    }
}

// === Пути ===
$storageDir = __DIR__ . '/storage';
$modelFile  = "$storageDir/category_model.phpml";

if (!file_exists($modelFile)) {
    die("❌ Модель не найдена. Сначала запусти category_trainer_SVC.php\n");
}

// === Загружаем модель (теперь NGramTokenizer уже объявлен!) ===
$modelManager = new ModelManager();
/** @var Pipeline $pipeline */
$pipeline = $modelManager->restoreFromFile($modelFile);

// === Получаем текст из CLI ===
if ($argc < 2) {
    die("Использование: php category_predict_SVC2.php \"текст для классификации\"\n");
}
$query = trim($argv[1]);

// === Получаем топ-N категорий с вероятностями через пайплайн ===
if (method_exists($pipeline, 'predictProbability')) {
    $probs = $pipeline->predictProbability([$query])[0];
    arsort($probs);

    $topN = 3;
    echo "🔍 Текст: $query\n";
    echo "📊 Топ-$topN категорий с вероятностями:\n";
    foreach (array_slice($probs, 0, $topN, true) as $cat => $p) {
        printf("  - %s: %.2f\n", $cat, $p);
    }
} else {
    $pred = $pipeline->predict([$query])[0];
    echo "🔍 Текст: $query\n";
    echo "📂 Категория: $pred\n";
}
