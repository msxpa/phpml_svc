#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';
require __DIR__ . '/CustomNaiveBayes.php';

use Phpml\Tokenization\WhitespaceTokenizer;
use Phpml\FeatureExtraction\TokenCountVectorizer;

$storageDir   = __DIR__ . '/storage';
$modelFile    = "$storageDir/category_model.php";
$trainingFile = "$storageDir/training_data.json"; // обычный JSON массив
$usedDir      = "$storageDir/used";
$batchSize    = 50000; // размер батча

// создаём папки
if (!is_dir($storageDir)) mkdir($storageDir, 0777, true);
if (!is_dir($usedDir)) mkdir($usedDir, 0777, true);

if (!file_exists($trainingFile)) {
    file_put_contents($trainingFile, "");
    echo "⚠️ training_data.json не найден. Создан пустой файл.\n";
    exit(0);
}

echo "📚 Начинаем обучение из $trainingFile\n";

// ---- 1. Построение словаря токенов ----
echo "🔹 Строим словарь токенов...\n";

$tokenizer = new WhitespaceTokenizer();
$vectorizer = new TokenCountVectorizer($tokenizer);

$handle = fopen($trainingFile, 'r');
if (!$handle) die("❌ Не удалось открыть файл $trainingFile\n");

$allSamples = [];
$labels = [];
$lineCount = 0;
$firstLine = true;

while (($line = fgets($handle)) !== false) {
    $line = trim($line);

    if ($firstLine) {
        $firstLine = false;
        if ($line === '[') continue; // пропускаем [
    }

    if ($line === ']') continue; // пропускаем ]

    // убираем запятую в конце
    $line = rtrim($line, ',');

    $item = json_decode($line, true);
    if (isset($item['title']) && isset($item['category'])) {
        $allSamples[] = mb_strtolower(trim($item['title']));
        $labels[]     = trim($item['category']);
        $lineCount++;
    }

    if ($lineCount % 100000 == 0) echo "  Прочитано $lineCount строк...\n";
}

fclose($handle);

echo "✅ Всего прочитано $lineCount объектов\n";

$vectorizer->fit($allSamples);
$vectorizer->transform($allSamples);
unset($allSamples); // освобождаем память

echo "✅ Словарь токенов построен, размер: " . count($vectorizer->getVocabulary()) . "\n";

// ---- 2. Инкрементальное обучение CustomNaiveBayes ----
$nb = new CustomNaiveBayes();

$handle = fopen($trainingFile, 'r');
if (!$handle) die("❌ Не удалось открыть файл $trainingFile\n");

$batchSamples = [];
$batchLabels  = [];
$lineCount = 0;
$firstLine = true;

while (($line = fgets($handle)) !== false) {
    $line = trim($line);

    if ($firstLine) {
        $firstLine = false;
        if ($line === '[') continue;
    }

    if ($line === ']') continue;
    $line = rtrim($line, ',');

    $item = json_decode($line, true);
    if (isset($item['title']) && isset($item['category'])) {
        $batchSamples[] = mb_strtolower(trim($item['title']));
        $batchLabels[]  = trim($item['category']);
        $lineCount++;
    }

    if (count($batchSamples) >= $batchSize) {
        $vectorizer->transform($batchSamples);
        $nb->partialFit($batchSamples, $batchLabels); // метод partialFit должен быть реализован
        $batchSamples = [];
        $batchLabels = [];
        echo "  Пройдено $lineCount строк...\n";
    }
}

// Последний батч
if (!empty($batchSamples)) {
    $vectorizer->transform($batchSamples);
    $nb->partialFit($batchSamples, $batchLabels);
}

fclose($handle);

echo "✅ Обучение завершено, всего обработано $lineCount строк\n";

// ---- 3. Сохраняем модель ----
file_put_contents($modelFile, serialize([
    'vectorizer' => $vectorizer,
    'nb'         => $nb
]));

echo "✅ Модель сохранена: $modelFile\n";

// ---- 4. Перемещаем JSON в used/ ----
$timestamp = date('Ymd_His');
$usedFile = "$usedDir/training_data_$timestamp.json";
rename($trainingFile, $usedFile);
file_put_contents($trainingFile, "");
echo "📦 Данные перемещены в: $usedFile\n";
echo "🧹 training_data.json сброшен\n";
echo "✨ Готово.\n";
