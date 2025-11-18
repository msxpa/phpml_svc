#!/usr/bin/env php
<?php
require __DIR__ . '/vendor/autoload.php'; // автозагрузка Composer
require __DIR__ . '/EnjoyNaiveBayes.php'; // наш классификатор

use Phpml\Tokenization\WhitespaceTokenizer;

// Папки и файлы
$storageDir   = __DIR__ . '/storage';
$modelFile    = "$storageDir/category_model.php"; // файл модели
$trainingFile = "$storageDir/training_data.json"; // JSON массив документов
$usedDir      = "$storageDir/used";
$vocabSize    = 500000;    // топ-N токенов
$saveEvery    = 500000;    // каждые N строк сохранять промежуточную модель

// Создаём папки, если их нет
if (!is_dir($storageDir)) mkdir($storageDir, 0777, true);
if (!is_dir($usedDir)) mkdir($usedDir, 0777, true);

// Если файл с обучением пустой — создаём пустой и выходим
if (!file_exists($trainingFile)) { file_put_contents($trainingFile, ""); exit(0); }

echo "📚 Начинаем обучение из $trainingFile\n";

// ---- 1. Сбор словаря токенов топ-N ----
echo "🔹 Сбор словаря токенов...\n";
$tokenizer = new WhitespaceTokenizer();
$tokenCounts = [];      // [token => count]
$lineCount = 0;
$batchSizeForVocab = 10000;
$batchTitles = [];

$handle = fopen($trainingFile, 'r');
$firstLine = true;

while (($line = fgets($handle)) !== false) {
    $line = trim($line);
    if ($firstLine) { $firstLine = false; if ($line === '[') continue; } // игнорируем '['
    if ($line === ']') continue; // игнорируем ']'
    $line = rtrim($line, ',');

    $item = json_decode($line, true);
    if (!isset($item['title'])) continue;

    // приводим к нижнему регистру и собираем батч
    $batchTitles[] = mb_strtolower(trim($item['title']));
    $lineCount++;

    if (count($batchTitles) >= $batchSizeForVocab) {
        // считаем токены
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

// оставляем только топ-N токенов
arsort($tokenCounts);
$tokenCounts = array_slice($tokenCounts, 0, $vocabSize, true);
$vocab = array_flip(array_keys($tokenCounts)); // token => index
$vecSize = count($vocab);
echo "✅ Словарь токенов построен, размер: $vecSize\n";

// ---- 2. Ultra-light обучение с промежуточным сохранением ----
echo "🔹 Начало обучения...\n";
$nb = new EnjoyNaiveBayes();

// Функция преобразования текста в sparse vector
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

    // Преобразуем в sparse vector и обучаем одну строку
    $vector = tokensToVector(mb_strtolower(trim($item['title'])), $vocab, $tokenizer);
    $nb->partialFit([$vector], [trim($item['category'])]);

    $lineCount++;
    if ($lineCount % 10000 == 0) echo "  Обучено $lineCount строк...\n";

    // Промежуточное сохранение модели
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

// ---- 3. Финальное сохранение модели ----
file_put_contents($modelFile, serialize([
    'vocab' => $vocab,
    'nb'    => $nb
]));
echo "✅ Финальная модель сохранена: $modelFile\n";

// ---- 4. Перемещаем исходный JSON в used ----
$timestamp = date('Ymd_His');
$usedFile = "$usedDir/training_data_$timestamp.json";
rename($trainingFile, $usedFile);
file_put_contents($trainingFile, "");
echo "📦 Данные перемещены в: $usedFile\n";
echo "✨ Готово.\n";
