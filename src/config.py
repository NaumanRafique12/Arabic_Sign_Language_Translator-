# src/config.py

# --- Model Configuration ---
THRESHOLD = 0.98
SEQUENCE_LENGTH = 30
KEYPOINTS_NUM = 84  # 42 landmarks * 2 coordinates (x, y)

# --- Actions List ---
# This MUST match the exact order of folders in your training data
ACTIONS = [
    'Act', 'Alhamdullah', 'all', 'Baba', 'Basis', 'because', 'Boy', 'Brave', 
    'calls', 'calm', 'College', 'Confused', 'conversation', 'Crying', 'daily', 
    'danger', 'disagree with', 'drinks', 'Eats', 'effort', 'Egypt', 'Enters', 
    'Excelent', 'explanation', 'Fasting', 'Female', 'First', 'For Us', 'fuel', 
    'Gift', 'Girl', 'Glass', 'good bye', 'government', 'Happy', 'hates', 'hears', 
    'Help', 'Here You Are', 'how_are_u', 'Humanity', 'hungry', 'I', 'ignorance', 
    'immediately', 'Important', 'Intelligent', 'Last', 'leader', 'Liar', 'Loves', 
    'male', 'Mama', 'memory', 'model', 'mostly', 'motive', 'Muslim', 'must', 
    'my home land', 'no', 'nonsense', 'obvious', 'Old', 'Palestine', 'prevent', 
    'ready', 'rejection', 'Right', 'selects', 'shut up', 'Sing', 'sleeps', 
    'smells', 'Somkes', 'Spoon', 'Summer', 'symposium', 'Tea', 'teacher', 
    'terrifying', 'Thanks', 'time', 'to boycott', 'to cheer', 'to go', 'to live', 
    'to spread', 'Toilet', 'trap', 'University', 'ur_welcome', 'victory', 
    'walks', 'Where', 'wheres_ur_house', 'Window', 'Winter', 'yes', 'You'
]

# --- Arabic Translation Mapping ---
# Maps the English folder names to Arabic display text
ARABIC_MAPPING = {
    'Act': 'تتصرف', 'Alhamdullah': 'الحمدلله', 'all': 'جميعا', 'Baba': 'بابا',
    'Basis': 'اساسيات', 'because': 'بسبب', 'Boy': 'ولد', 'Brave': 'شجاع',
    'calls': 'اتصل', 'calm': 'هادئ', 'College': 'كليه', 'Confused': 'متوتر',
    'conversation': 'محادثه', 'Crying': 'بكاء', 'daily': 'يوميا', 'danger': 'خطر',
    'disagree with': 'اختلف مع', 'drinks': 'اشرب', 'Eats': 'اكل', 'effort': 'مجهود',
    'Egypt': 'مصر', 'Enters': 'دخول', 'Excelent': 'ممتاز', 'explanation': 'الشرح',
    'Fasting': 'في الصيام', 'Female': 'انثي', 'First': 'اولا', 'For Us': 'لنا',
    'fuel': 'الوقود', 'Gift': 'الهديه', 'Girl': 'البنت', 'Glass': 'الكوب',
    'good bye': 'الي اللقاء', 'government': 'الحكومه', 'Happy': 'بسعاده',
    'hates': 'اكره', 'hears': 'اسمع', 'Help': 'تساعد', 'Here You Are': 'تفضل',
    'how_are_u': 'كيف حالك', 'Humanity': 'البشريه', 'hungry': 'اشعر بالجوع',
    'I': 'انا', 'ignorance': 'تجاهل', 'immediately': 'حالا', 'Important': 'المهم',
    'Intelligent': 'الذكاء', 'Last': 'الاخير', 'leader': 'القائد', 'Liar': 'الكاذب',
    'Loves': 'احب', 'male': 'ذكر', 'Mama': 'ماما', 'memory': 'الذاكره',
    'model': 'نموذج', 'mostly': 'في الغالب', 'motive': 'الدافع', 'Muslim': 'مسلم',
    'must': 'لازم', 'my home land': 'وطني', 'no': 'لا', 'nonsense': 'الهراء',
    'obvious': 'بديهي', 'Old': 'القديمه', 'Palestine': 'فلسطين', 'prevent': 'اوقف',
    'ready': 'جاهز_ل', 'rejection': 'الرفض', 'Right': 'صحيح', 'selects': 'تختار',
    'shut up': 'اصمت', 'Sing': 'الغني', 'sleeps': 'النوم', 'smells': 'اشم',
    'Somkes': 'يسمح بالتدخين', 'Spoon': 'المعلقه', 'Summer': 'الصيف',
    'symposium': 'الندوه', 'Tea': 'الشاي', 'teacher': 'معلمي', 'terrifying': 'مخيف',
    'Thanks': 'شكرا لكم', 'time': 'الوقت', 'to boycott': 'تقاطع لاجل',
    'to cheer': 'تهتف ل', 'to go': 'ذاهب الي', 'to live': 'ل اعيش',
    'to spread': 'تنتشر', 'Toilet': 'دوره المياه', 'trap': 'فخ',
    'University': 'الجامعه', 'ur_welcome': 'مرحباً بكم', 'victory': 'النصر',
    'walks': 'المشي', 'Where': 'اين', 'wheres_ur_house': 'اين تسكن',
    'Window': 'الشباك', 'Winter': 'الشتاء', 'yes': 'نعم', 'You': 'انت'
}