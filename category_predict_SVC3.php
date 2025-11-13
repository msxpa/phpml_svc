<?php
require __DIR__ . '/vendor/autoload.php';

use Phpml\Classification\SVC;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\ModelManager;
use Phpml\Tokenization\Tokenizer;

// === Кастомный токенизатор ===
class NGramTokenizer implements Tokenizer {
    private int $n;
    public function __construct(int $n = 2) { $this->n = $n; }
    public function tokenize(string $text): array {
        $words = preg_split('/\s+/u', mb_strtolower($text));
        $ngrams = [];
        for ($i = 0; $i < count($words); $i++) {
            $ngrams[] = $words[$i];
            if ($i < count($words) - 1 && $this->n >= 2) {
                $ngrams[] = $words[$i] . ' ' . $words[$i + 1];
            }
        }
        return $ngrams;
    }
}

// === Пути ===
$storageDir = __DIR__ . '/storage';
$modelFile  = "$storageDir/category_model.phpml";
$vectorFile = "$storageDir/vectorizer.phpml";

if (!file_exists($modelFile) || !file_exists($vectorFile)) {
    die("❌ Модель или векторизатор не найдены. Сначала запусти trainer.\n");
}

// === Загружаем модель и векторизатор ===
$modelManager = new ModelManager();
/** @var SVC $classifier */
$classifier = $modelManager->restoreFromFile($modelFile);
/** @var TokenCountVectorizer $vectorizer */
$vectorizer = unserialize(file_get_contents($vectorFile));

// === CLI input ===
if ($argc < 2) {
    die("Использование: php category_predict_SVC3.php \"текст для классификации\"\n");
}
$query = mb_strtolower(trim($argv[1]));

// === Векторизация ===
$sample = [$query];
$vectorizer->transform($sample);

// === Получаем вероятности категорий ===
$probs = $classifier->predictProbability($sample)[0];
arsort($probs);

// === Вывод топ-3 категорий ===
$topN = 3;
echo "🔍 Текст: $query\n";
echo "📊 Топ-$topN категорий с вероятностями:\n";
foreach (array_slice($probs, 0, $topN, true) as $cat => $p) {
    printf("  - %s: %.2f\n", $cat, $p);
}
