#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php';
require __DIR__ . '/EnjoyNaiveBayes.php';

use Phpml\Tokenization\WhitespaceTokenizer;

$storageDir   = __DIR__ . '/storage';
$modelFile    = "$storageDir/category_model.php";
$trainingFile = "$storageDir/training_data.json";
$usedDir      = "$storageDir/used";
$vocabSize    = 500000;
$saveEvery    = 500000; // сохранять промежуточную модель каждые N строк

if (!is_dir($storageDir)) mkdir($storageDir, 0777, true);
if (!is_dir($usedDir)) mkdir($usedDir, 0777, true);
if (!file_exists($trainingFile)) { file_put_contents($trainingFile, ""); exit(0); }

echo "📚 Начинаем обучение из $trainingFile\n";

// ---- 1. Строим словарь токенов топ-N на лету ----
echo "🔹 Сбор словаря токенов...\n";
$tokenizer = new WhitespaceTokenizer();
$tokenCounts = [];
$lineCount = 0;
$batchSizeForVocab = 10000;
$batchTitles = [];

$handle = fopen($trainingFile, 'r');
$firstLine = true;

while (($line = fgets($handle)) !== false) {
    $line = trim($line);
    if ($firstLine) { $firstLine = false; if ($line === '[') continue; }
    if ($line === ']') continue;
    $line = rtrim($line, ',');

    $item = json_decode($line, true);
    if (!isset($item['title'])) continue;

    $batchTitles[] = mb_strtolower(trim($item['title']));
    $lineCount++;

    if (count($batchTitles) >= $batchSizeForVocab) {
        foreach ($batchTitles as $t) {
            foreach ($tokenizer->tokenize($t) as $token) {
                $tokenCounts[$token] = ($tokenCounts[$token] ?? 0) + 1;
            }
        }
        $batchTitles = [];
    }

    if ($lineCount % 100000 == 0) echo "  Прочитано $lineCount строк для словаря...\n";
}

// последний батч
if (!empty($batchTitles)) {
    foreach ($batchTitles as $t) {
        foreach ($tokenizer->tokenize($t) as $token) {
            $tokenCounts[$token] = ($tokenCounts[$token] ?? 0) + 1;
        }
    }
}
fclose($handle);

// топ-N токенов
arsort($tokenCounts);
$tokenCounts = array_slice($tokenCounts, 0, $vocabSize, true);
$vocab = array_flip(array_keys($tokenCounts));
$vecSize = count($vocab);
echo "✅ Словарь токенов построен, размер: $vecSize\n";

// ---- 2. Ultra-light обучение с промежуточным сохранением ----
echo "🔹 Начало обучения...\n";
$nb = new EnjoyNaiveBayes();

function tokensToVector(string $text, array $vocab, WhitespaceTokenizer $tokenizer): array {
    $vec = [];
    foreach ($tokenizer->tokenize($text) as $token) {
        if (isset($vocab[$token])) $vec[$vocab[$token]] = ($vec[$vocab[$token]] ?? 0) + 1;
    }
    return $vec;
}

$handle = fopen($trainingFile, 'r');
$firstLine = true;
$lineCount = 0;

while (($line = fgets($handle)) !== false) {
    $line = trim($line);
    if ($firstLine) { $firstLine = false; if ($line === '[') continue; }
    if ($line === ']') continue;
    $line = rtrim($line, ',');

    $item = json_decode($line, true);
    if (!isset($item['title'], $item['category'])) continue;

    $vector = tokensToVector(mb_strtolower(trim($item['title'])), $vocab, $tokenizer);
    $nb->partialFit([$vector], [trim($item['category'])]);

    $lineCount++;
    if ($lineCount % 10000 == 0) echo "  Обучено $lineCount строк...\n";

    // Промежуточное сохранение
    if ($lineCount % $saveEvery == 0) {
        file_put_contents($modelFile . '.tmp', serialize([
            'vocab' => $vocab,
            'nb'    => $nb
        ]));
        echo "💾 Промежуточная модель сохранена после $lineCount строк...\n";
    }
}
fclose($handle);

echo "✅ Обучение завершено, всего обработано $lineCount строк\n";

// ---- 3. Сохраняем финальную модель ----
file_put_contents($modelFile, serialize([
    'vocab' => $vocab,
    'nb'    => $nb
]));
echo "✅ Финальная модель сохранена: $modelFile\n";

// ---- 4. Перемещаем JSON в used ----
$timestamp = date('Ymd_His');
$usedFile = "$usedDir/training_data_$timestamp.json";
rename($trainingFile, $usedFile);
file_put_contents($trainingFile, "");
echo "📦 Данные перемещены в: $usedFile\n";
echo "✨ Готово.\n";
