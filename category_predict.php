#!/usr/bin/env php
<?php
/**
 * Category Predictor
 *
 * Использует обученную модель category_model.phpml
 * Принимает аргумент (title) и выдаёт предсказанную категорию
 *
 * Использование:
 *   CLI: php category_predict.php "Wooden lamp modern"
 *   HTTP: ?title=Wooden lamp modern
 */

require __DIR__ . '/vendor/autoload.php';

use Phpml\ModelManager;

// === Пути ===
$modelFile = __DIR__ . '/storage/category_model.phpml';

// === Проверка наличия модели ===
if (!file_exists($modelFile)) {
    echo "❌ Модель не найдена. Сначала обучите её с помощью category_trainer.php.\n";
    exit(1);
}

// === Восстанавливаем модель ===
$modelManager = new ModelManager();
$pipeline = $modelManager->restoreFromFile($modelFile);

// === Получаем title ===
if (php_sapi_name() === 'cli') {
    // Режим CLI
    $input = $argv[1] ?? null;
} else {
    // Режим Web
    $input = $_GET['title'] ?? $_POST['title'] ?? null;
}

if (empty($input)) {
    echo "⚠️ Не указан title.\n";
    echo "Пример: php category_predict.php \"Wooden lamp\"\n";
    exit(0);
}

// === Подготовка текста ===
$title = mb_strtolower(trim($input));
$title = preg_replace('/[^a-zа-я0-9\s]+/ui', '', $title);

try {
    $predicted = $pipeline->predict($title);
    echo "🔍 Title: $input\n";
    echo "📂 Категория: $predicted\n";
} catch (Exception $e) {
    echo "⚠️ Ошибка при предсказании: " . $e->getMessage() . "\n";
    exit(1);
}
