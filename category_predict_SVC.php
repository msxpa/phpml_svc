#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';

use Phpml\ModelManager;
use Phpml\Pipeline;

// === Определяем кастомный токенизатор перед восстановлением модели ===
use Phpml\Tokenization\Tokenizer;

class NGramTokenizer implements Tokenizer {
    private int $n;

    public function __construct(int $n = 2) {
        $this->n = $n;
    }

    public function tokenize(string $text): array {
        $words = preg_split('/\s+/u', mb_strtolower($text));
        $ngrams = [];
        $count = count($words);

        for ($i = 0; $i < $count; $i++) {
            $ngrams[] = $words[$i]; // униграмма
            if ($i < $count - 1 && $this->n >= 2) {
                $ngrams[] = $words[$i] . ' ' . $words[$i + 1]; // биграмма
            }
        }
        return $ngrams;
    }
}

// === Пути ===
$storageDir = __DIR__ . '/storage';
$modelFile  = "$storageDir/category_model.phpml";

if (!file_exists($modelFile)) {
    die("❌ Модель не найдена. Сначала запусти category_trainer.php\n");
}

$modelManager = new ModelManager();
/** @var Pipeline $pipeline */
$pipeline = $modelManager->restoreFromFile($modelFile);

// Получаем текст из аргумента CLI
if ($argc < 2) {
    die("Использование: php multi_predict.php \"текст для классификации\"\n");
}

$query = mb_strtolower(trim($argv[1]));

// Прогноз категории
try {
    $predicted = $pipeline->predict([$query])[0];
    echo "🔍 Текст: $query\n";
    echo "📂 Категория: $predicted\n";
} catch (Exception $e) {
    echo "⚠️ Ошибка при предсказании: " . $e->getMessage() . "\n";
}
