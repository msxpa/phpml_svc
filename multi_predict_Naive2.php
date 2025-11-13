#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';

use Phpml\ModelManager;
use Phpml\Classification\NaiveBayes;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Pipeline;

// Пути
$storageDir = __DIR__ . '/storage';
$modelFile  = "$storageDir/category_model.phpml";

if (!file_exists($modelFile)) {
    die("❌ Модель не найдена. Сначала запусти category_trainer.php\n");
}

// Загружаем модель
$modelManager = new ModelManager();
/** @var Pipeline $pipeline */
$pipeline = $modelManager->restoreFromFile($modelFile);

// Получаем текст из аргумента CLI
if ($argc < 2) {
    die("Использование: php multi_predict_Naive.php \"текст для классификации\"\n");
}

$query = mb_strtolower(trim($argv[1]));
$query = preg_replace('/[^a-zа-я0-9\s]+/ui', '', $query);

echo "🔍 Текст: $query\n";

// Достаем классификатор и векторизатор
$refPipeline = new ReflectionObject($pipeline);
$estimatorProp = $refPipeline->getProperty('estimator');
$estimatorProp->setAccessible(true);
$classifier = $estimatorProp->getValue($pipeline);

$transformersProp = $refPipeline->getProperty('transformers');
$transformersProp->setAccessible(true);
$transformers = $transformersProp->getValue($pipeline);
$vectorizer = $transformers[0];

// Преобразуем текст в вектор
$vectorizer->fit([$query]); // нужно только если новые слова
//$vectorized = $vectorizer->transform([$query]);
$sampleArray = [$query];       // создаём переменную, а не временный массив
$vectorized = $vectorizer->transform($sampleArray);


if ($classifier instanceof NaiveBayes) {
    $probs = $classifier->predictProbability($vectorized)[0];
    arsort($probs);

    echo "📊 Вероятности категорий:\n";
    foreach ($probs as $cat => $p) {
        printf("  - %s: %.2f\n", $cat, $p);
    }

    // Вывод топ-3 категорий
    $topN = 3;
    echo "\n📂 Топ-$topN категорий: " . implode(', ', array_slice(array_keys($probs), 0, $topN)) . "\n";
} else {
    $pred = $pipeline->predict([$query])[0];
    echo "📂 Категория: $pred\n";
}
