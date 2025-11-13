#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';

use Phpml\Classification\SVC;
use Phpml\SupportVectorMachine\Kernel;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\Tokenizer;
use Phpml\ModelManager;

// === Кастомный токенизатор для униграмм + биграмм ===
class NGramTokenizer implements Tokenizer {
    private int $n;
    public function __construct(int $n = 2) { $this->n = $n; }
    public function tokenize(string $text): array {
        $words = preg_split('/\s+/u', mb_strtolower($text));
        $ngrams = [];
        for ($i = 0; $i < count($words); $i++) {
            $ngrams[] = $words[$i]; // униграмма
            if ($i < count($words) - 1 && $this->n >= 2) {
                $ngrams[] = $words[$i] . ' ' . $words[$i + 1]; // биграмма
            }
        }
        return $ngrams;
    }
}

// === Пути ===
$storageDir   = __DIR__ . '/storage';
$modelFile    = "$storageDir/category_model.phpml";
$vectorFile   = "$storageDir/vectorizer.phpml";
$trainingFile = "$storageDir/training_data.json";
$usedDir      = "$storageDir/used";

if (!is_dir($storageDir)) mkdir($storageDir, 0777, true);
if (!is_dir($usedDir)) mkdir($usedDir, 0777, true);

$modelManager = new ModelManager();

// === Загружаем данные для обучения ===
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

// === Создаём векторизатор ===
$vectorizer = new TokenCountVectorizer(new NGramTokenizer());
$vectorizer->fit($samples);
$vectorizer->transform($samples);

// === Создаём классификатор SVC с вероятностями ===
$classifier = new SVC(
    Kernel::LINEAR,
    1000,
    3,
    null,
    0.0,
    0.001,
    100,
    true,
    true // probabilityEstimates
);

// === Обучаем модель ===
$classifier->train($samples, $labels);

// === Сохраняем модель и векторизатор ===
//$modelManager->saveToFile($classifier, $modelFile);
//$modelManager->saveToFile($vectorizer, $vectorFile);

// === Сохраняем модель ===
$modelManager->saveToFile($classifier, $modelFile);

// === Сохраняем векторизатор через serialize ===
file_put_contents($vectorFile, serialize($vectorizer));

// === Переносим JSON в used/ ===
$timestamp = date('Ymd_His');
$usedFile = "$usedDir/training_data_$timestamp.json";
rename($trainingFile, $usedFile);
file_put_contents($trainingFile, "[]");

echo "✅ Модель и векторизатор обновлены.\n";
echo "📦 Данные перенесены в: $usedFile\n";
