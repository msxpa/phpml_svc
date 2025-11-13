#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';

use Phpml\Classification\NaiveBayes;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\WhitespaceTokenizer;
use Phpml\Pipeline;
use Phpml\ModelManager;

$storageDir   = __DIR__ . '/storage';
$modelFile    = "$storageDir/category_model.phpml";
$trainingFile = "$storageDir/training_data.json";
$usedDir      = "$storageDir/used";

if (!is_dir($storageDir)) mkdir($storageDir, 0777, true);
if (!is_dir($usedDir)) mkdir($usedDir, 0777, true);

$modelManager = new ModelManager();

// Создаём Pipeline с векторизатором и NaiveBayes
$vectorizer = new TokenCountVectorizer(new WhitespaceTokenizer());
$classifier = new NaiveBayes();
//$pipeline = new Pipeline([$vectorizer, $classifier]);
$pipeline = new Pipeline([$vectorizer], $classifier);

// Загружаем данные
if (!file_exists($trainingFile)) {
    file_put_contents($trainingFile, "[]");
    echo "⚠️ training_data.json не найден. Создан пустой файл.\n";
    exit(0);
}

$data = json_decode(file_get_contents($trainingFile), true);
if (empty($data) || !is_array($data)) {
    echo "⚠️ training_data.json пуст или имеет неверный формат.\n";
    exit(0);
}

// Преобразуем данные
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

// Обучаем Pipeline
$pipeline->train($samples, $labels);

// Сохраняем модель
$modelManager->saveToFile($pipeline, $modelFile);
echo "✅ Модель обновлена и сохранена: $modelFile\n";

// Переносим JSON в used/
$timestamp = date('Ymd_His');
$usedFile = "$usedDir/training_data_$timestamp.json";
rename($trainingFile, $usedFile);
file_put_contents($trainingFile, "[]");
echo "📦 Данные перенесены в: $usedFile\n";
echo "🧹 training_data.json сброшен.\n";
echo "✅ Готово.\n";
