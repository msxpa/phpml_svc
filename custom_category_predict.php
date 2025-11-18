#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';
require __DIR__ . '/CustomNaiveBayes.php';

$storageDir = __DIR__ . '/storage';
$modelFile  = "$storageDir/category_model.php";

if (!file_exists($modelFile)) {
    die("❌ Модель не найдена. Сначала запусти trainer.\n");
}

$modelData = unserialize(file_get_contents($modelFile));
/** @var Phpml\FeatureExtraction\TokenCountVectorizer $vectorizer */
$vectorizer = $modelData['vectorizer'];
/** @var CustomNaiveBayes $nb */
$nb = $modelData['nb'];

if ($argc < 2) {
    die("Использование: php category_predict.php \"текст\"\n");
}

$query = mb_strtolower(trim($argv[1]));
$sample = [$query];

// ---- Transform text to vector ----
$vectorizer->transform($sample);
$vector = $sample[0];

// ---- Predict top-3 ----
$top3 = $nb->predictTopN([$vector], 3);

echo "🔍 Текст: $query\n";
echo "📊 Топ-3 категорий:\n";

foreach ($top3[0] as $cat => $prob) {
    printf("  - %s: %.4f\n", $cat, $prob);
}
