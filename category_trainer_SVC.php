#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';

use Phpml\Classification\SVC;
use Phpml\SupportVectorMachine\Kernel;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\Tokenizer;
use Phpml\Pipeline;
use Phpml\ModelManager;

// === Кастомный токенизатор для униграмм + биграмм ===
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
$storageDir   = __DIR__ . '/storage';
$modelFile    = "$storageDir/category_model.phpml";
$trainingFile = "$storageDir/training_data.json";
$usedDir      = "$storageDir/used";

// === Создание папок если нужно ===
if (!is_dir($storageDir)) mkdir($storageDir, 0777, true);
if (!is_dir($usedDir)) mkdir($usedDir, 0777, true);

$modelManager = new ModelManager();

// === Загружаем или создаём модель ===
if (file_exists($modelFile)) {
    echo "🔁 Загрузка обученной модели...\n";
    $pipeline = $modelManager->restoreFromFile($modelFile);
} else {
    echo "🧠 Модель не найдена, создаём новую...\n";
    $vectorizer = new TokenCountVectorizer(new NGramTokenizer());
    $classifier = new SVC(Kernel::LINEAR, 1000);
    $pipeline = new Pipeline([$vectorizer], $classifier);
}

// === Загружаем новые данные ===
if (!file_exists($trainingFile)) {
    echo "⚠️ training_data.json не найден. Создаём пустой...\n";
    file_put_contents($trainingFile, "[]");
    exit(0);
}

$data = json_decode(file_get_contents($trainingFile), true);

if (empty($data) || !is_array($data)) {
    echo "⚠️ training_data.json пуст или некорректен.\n";
    exit(0);
}

// === Преобразуем данные ===
$samples = [];
$labels  = [];
foreach ($data as $item) {
    if (!empty($item['title']) && !empty($item['category'])) {
        $samples[] = mb_strtolower(trim($item['title']));
        $labels[]  = trim($item['category']);
    }
}

if (empty($samples)) {
    echo "⚠️ Нет валидных примеров для обучения.\n";
    exit(0);
}

echo "📚 Обучение на " . count($samples) . " примерах...\n";

// === Обучение модели ===
$pipeline->train($samples, $labels);
$modelManager->saveToFile($pipeline, $modelFile);

echo "✅ Модель обновлена и сохранена: $modelFile\n";

// === Переносим использованный JSON в used/ ===
$timestamp = date('Ymd_His');
$usedFile = "$usedDir/training_data_$timestamp.json";
rename($trainingFile, $usedFile);
file_put_contents($trainingFile, "[]");

echo "📦 Данные перенесены в: $usedFile\n";
echo "🧹 training_data.json сброшен.\n";
echo "✅ Готово.\n";
