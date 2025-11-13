#!/usr/bin/env php
<?php
/**
 * Category Classifier with Feedback (v2)
 *
 * Функционал:
 *  - Загружает обученные данные из storage/training_data.json
 *  - Обучает/дообучает модель php-ml
 *  - Переносит использованный JSON в storage/used/
 *  - Сохраняет пустой training_data.json после обучения
 *  - Позволяет интерактивно классифицировать и добавлять новые примеры
 */

require __DIR__ . '/vendor/autoload.php';

use Phpml\Classification\NaiveBayes;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\Tokenization\WhitespaceTokenizer;
use Phpml\Pipeline;
use Phpml\ModelManager;

// === Пути ===
$storageDir = __DIR__ . '/storage';
$modelFile = $storageDir . '/category_model.phpml';
$trainingFile = $storageDir . '/training_data.json';
$usedDir = $storageDir . '/used';

// === Проверяем и создаём папки ===
if (!is_dir($storageDir)) mkdir($storageDir, 0777, true);
if (!is_dir($usedDir)) mkdir($usedDir, 0777, true);

// === Инициализация менеджера модели ===
$modelManager = new ModelManager();
$pipeline = null;

// === Загружаем или создаём модель ===
if (file_exists($modelFile)) {
    echo "🔁 Загрузка обученной модели...\n";
    $pipeline = $modelManager->restoreFromFile($modelFile);
} else {
    echo "🧠 Модель не найдена, создаём новую...\n";
    $vectorizer = new TokenCountVectorizer(new WhitespaceTokenizer());
    $classifier = new NaiveBayes();
    // ❌ Было: $pipeline = new Pipeline([$vectorizer, $classifier]);
    // ✅ Стало:
    $pipeline = new Pipeline([$vectorizer], $classifier);
}

// === Загружаем новые данные ===
if (file_exists($trainingFile)) {
    $raw = file_get_contents($trainingFile);
    $raw = trim($raw);
    $data = json_decode($raw, true);

    if (!empty($data) && is_array($data)) {
        // Если json в виде [{title:..., category:...}, ...]
        $samples = [];
        $labels = [];

        foreach ($data as $item) {
            if (!empty($item['title']) && !empty($item['category'])) {
                $samples[] = mb_strtolower(trim($item['title']));
                $labels[] = trim($item['category']);
            }
        }

        if (!empty($samples)) {
            echo "📚 Найдено " . count($samples) . " новых примеров. Обновляем модель...\n";

            // Дообучаем модель
            $pipeline->train($samples, $labels);
            $modelManager->saveToFile($pipeline, $modelFile);

            echo "✅ Модель успешно обновлена.\n";

            // Перемещаем training_data.json в used/
            $timestamp = date('Ymd_His');
            $usedFile = "$usedDir/training_data_$timestamp.json";
            rename($trainingFile, $usedFile);
            echo "📦 Перемещено в: $usedFile\n";

            // Создаём пустой training_data.json
            file_put_contents($trainingFile, "{}");
            echo "🧹 Новый пустой training_data.json создан.\n";
        } else {
            echo "⚠️ training_data.json не содержит валидных примеров.\n";
        }
    } else {
        echo "⚠️ training_data.json пуст или имеет неверный формат.\n";
    }
} else {
    echo "⚠️ training_data.json не найден. Создаём пустой...\n";
    file_put_contents($trainingFile, "{}");
}

// === Основной интерактивный цикл ===
echo "\n=== Режим классификации ===\n";
echo "Введите поисковый запрос (или 'exit' для выхода)\n\n";

while (true) {
    $query = readline("🔎 Запрос: ");
    $query = mb_strtolower(trim($query));
    $query = preg_replace('/[^a-zа-я0-9\s]+/ui', '', $query);

    if ($query === 'exit') break;
    if (empty($query)) continue;

    try {
        $predicted = $pipeline->predict($query);
        echo "📂 Предсказанная категория: $predicted\n";
    } catch (Exception $e) {
        echo "⚠️ Модель пока не обучена.\n";
        $predicted = null;
    }

    $feedback = readline("Введите правильную категорию (или Enter, если верно): ");
    $category = !empty($feedback) ? $feedback : $predicted;

    if (empty($category)) {
        echo "⏭ Пропущено — нет категории.\n";
        continue;
    }

    echo "✅ Категория для '$query': $category\n";

    // === Добавляем в training_data.json ===
    $newExample = [
        'title' => $query,
        'category' => $category,
    ];

    $existingData = [];
    $rawJson = file_get_contents($trainingFile);
    $decoded = json_decode($rawJson, true);

    if (is_array($decoded)) {
        // JSON может быть {} — в этом случае просто создаём массив
        if (array_values($decoded) !== $decoded) {
            $existingData = [];
        } else {
            $existingData = $decoded;
        }
    }

    $existingData[] = $newExample;
    file_put_contents($trainingFile, json_encode($existingData, JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT));

    echo "💾 Пример добавлен в training_data.json (обучение произойдёт при следующем запуске)\n\n";
}

echo "👋 Завершение работы.\n";
