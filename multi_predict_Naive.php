#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';

use Phpml\ModelManager;
use Phpml\Pipeline;
use Phpml\Classification\NaiveBayes;

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
    die("Использование: php multi_predict.php \"текст для классификации\"\n");
}

$query = mb_strtolower(trim($argv[1]));
$query = preg_replace('/[^a-zа-я0-9\s]+/ui', '', $query);

echo "🔍 Текст: $query\n";

// Добираемся до классификатора внутри пайплайна
$reflection = new ReflectionObject($pipeline);
$prop = $reflection->getProperty('estimator');
$prop->setAccessible(true);
$classifier = $prop->getValue($pipeline);

if ($classifier instanceof NaiveBayes && method_exists($classifier, 'predictProbability')) {
    $vectorizer = $pipeline->getTransformers()[0];
    $vectorizer->fit([$query]); // на случай новых слов
    $vectorized = $vectorizer->transform([$query]);
    $probs = $classifier->predictProbability($vectorized)[0];

    arsort($probs);

    echo "📊 Вероятности категорий:\n";
    foreach ($probs as $cat => $p) {
        printf("  - %s: %.2f\n", $cat, $p);
    }

    // Пример: вывод категорий с вероятностью > 0.2
    $threshold = 0.2;
    $filtered = array_filter($probs, fn($v) => $v >= $threshold);
    echo "\n📂 Возможные категории (>$threshold): " . implode(', ', array_keys($filtered)) . "\n";

} else {
    // fallback
    var_dump($pipeline->predict([$query]));

    $pred = $pipeline->predict([$query])[0];
    echo "📂 Категория: $pred\n";
}
