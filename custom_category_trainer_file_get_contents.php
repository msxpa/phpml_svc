#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';
require __DIR__ . '/CustomNaiveBayes.php';

use Phpml\Tokenization\WhitespaceTokenizer;
use Phpml\FeatureExtraction\TokenCountVectorizer;

$storageDir   = __DIR__ . '/storage';
$modelFile    = "$storageDir/category_model.php";
$trainingFile = "$storageDir/training_data.json";
$usedDir      = "$storageDir/used";

// создаём папки
if (!is_dir($storageDir)) mkdir($storageDir, 0777, true);
if (!is_dir($usedDir)) mkdir($usedDir, 0777, true);

// загружаем данные
if (!file_exists($trainingFile)) {
    file_put_contents($trainingFile, "[]");
    echo "⚠️ training_data.json не найден. Создан пустой файл.\n";
    exit(0);
}

$data = json_decode(file_get_contents($trainingFile), true);
if (empty($data) || !is_array($data)) {
    echo "⚠️ training_data.json пуст или неверный формат.\n";
    exit(0);
}

$samples_raw = [];
$labels      = [];

foreach ($data as $item) {
    if (!empty($item['title']) && !empty($item['category'])) {
        $samples_raw[] = mb_strtolower(trim($item['title']));
        $labels[]      = trim($item['category']);
    }
}

if (empty($samples_raw)) {
    echo "⚠️ Нет валидных примеров для обучения.\n";
    exit(0);
}

echo "📚 Обучение на " . count($samples_raw) . " примерах...\n";

// ---- Vectorizer ----
$tokenizer = new WhitespaceTokenizer();
$vectorizer = new TokenCountVectorizer($tokenizer);
$vectorizer->fit($samples_raw);
$vectorizer->transform($samples_raw);

// ---- Train CustomNaiveBayes ----
$nb = new CustomNaiveBayes();
$nb->fit($samples_raw, $labels);

// ---- Save model ----
file_put_contents($modelFile, serialize([
    'vectorizer' => $vectorizer,
    'nb'         => $nb
]));

echo "✅ Модель сохранена: $modelFile\n";

// ---- Перемещаем training_data.json ----
$timestamp = date('Ymd_His');
$usedFile = "$usedDir/training_data_$timestamp.json";
rename($trainingFile, $usedFile);
file_put_contents($trainingFile, "[]");
echo "📦 Данные перенесены в: $usedFile\n";
echo "🧹 training_data.json сброшен.\n";
echo "✨ Готово.\n";
