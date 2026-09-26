# ── Sinonimlar (boyitilgan) ───────────────────────────────────────────────────

_SYNONYMS: dict[str, list[str]] = {
    # Jo'natish
    "yuborish":  ["jo'natish","yetkazish","pochta","topshirish","jo'natmoq","yubormoq","yetkazib berish"],
    "jonatish":  ["yuborish","yetkazish","pochta","topshirish","jo'natish"],
    "yetkazish": ["yuborish","jo'natish","topshirish","deliver","dostavka"],

    # Narx
    "qancha":    ["narxi","narx","turadi","tolov","сколько","цена","price","cost","fee","stoit","tarif","haq"],
    "narx":      ["qancha","tarif","turadi","тариф","price","cost","стоит","hisoblash","formula"],
    "tarif":     ["narx","qancha","price","тариф","rate","tolov","haq","stoimost"],
    "tolov":     ["narx","qancha","tarif","haq","summa","komissiya"],
    "turadi":    ["narx","qancha","tarif","stoit","стоит","costs"],

    # Tezlik
    "tez":       ["tezkor","ekspress","shoshilinch","срочно","express","urgent","quick","fast","bir kunda","1 kunda"],
    "tezkor":    ["tez","ekspress","bir kunda","быстро","express","urgent","shoshilinch"],
    "shoshilinch":["tez","tezkor","ekspress","срочно","urgent","express"],
    "ekspress":  ["tez","tezkor","ems","bir kunda","express","срочный"],

    # Saqlash
    "saqlash":   ["saqlanadi","muddati","kun","хранение","storage","turibdi","qolibdi","kutish"],
    "muddat":    ["saqlash","kun","срок","period","necha kun","qancha vaqt"],
    "bildirishnoma":["xabarnoma","извещение","notice","ogohlantirish","xabar"],

    # Mavjudlik
    "bormi":     ["mavjudmi","ishlaydi","есть ли","available","bor","bormи","qiladimi","qiladilarmi"],
    "mavjud":    ["bor","есть","available","ishlaydi","qilinadi"],

    # Gibrid pochta
    "gibrid":    ["gibrit","гибрид","hybrid","elektron hujjat","qog'ozda","chop etib","rasmiy xat"],
    "gibrit":    ["gibrid","гибрид","hybrid","elektron","qog'oz"],

    # Kuryer
    "kurierlik": ["kuryerlik","kuryer","courier","uyga keladi","uydan oladi"],
    "kurier":    ["kuryer","courier","курьер","kuryerlik"],
    "kuryer":    ["kurier","courier","курьер","kuryerlik","uyga yetkazish"],
    "kuryerlik": ["kurier","courier","курьер","kuryer"],

    # Shaharlar
    "samarqand": ["samarkand","самарканд","samarqand"],
    "nukus":     ["нукус","nukus","qoraqalpog'iston"],
    "termiz":    ["termez","термез","surxondaryo"],
    "urganch":   ["urgench","ургенч","xorazm"],
    "buxoro":    ["bukhara","бухара","buxoro"],
    "fargona":   ["fergana","фергана","farg'ona","farghona"],
    "andijon":   ["andijan","андижан"],
    "namangan":  ["наманган"],
    "qarshi":    ["karshi","карши","qashqadaryo"],
    "navoiy":    ["navoi","навои"],
    "jizzax":    ["jizzakh","джизак"],
    "toshkent":  ["ташкент","toshkent","tashkent"],

    # Posilka
    "posilka":   ["paket","посылка","parcel","yuk","tovar","buyum","jo'natma","paket"],
    "paket":     ["posilka","посылка","parcel","jo'natma"],

    # Xat
    "xat":       ["письмо","letter","hujjat","dokument","maktub","konvert"],
    "hujjat":    ["xat","document","letter","shartnoma","papers","qog'oz"],

    # Banderol
    "banderol":  ["бандероль","wrapper","kitob","jurnal","bosma","nashr"],
    "kitob":     ["banderol","book","нота","jurnal","дисс","dissertatsiya"],

    # Otkritka
    "otkritka":  ["открытка","postcard","tabriknoma","salomlashish","varaqcha"],
    "tabriknoma":["otkritka","открытка","postcard","greetings"],

    # Trek/kuzatish
    "trek":      ["kuzatish","отследить","tracking","raqam","qayerda","status","holat"],
    "kuzatish":  ["trek","tracking","отследить","raqam","status"],
    "raqam":     ["trek","tracking","номер","number","kod"],

    # Pul o'tkazma
    "pul":       ["otkazma","transfer","перевод","money","komissiya","yuborish","olish"],
    "otkazma":   ["pul","transfer","перевод","o'tkazma"],
    "komissiya": ["foiz","percent","tarif","haq","3%","6%","fee"],

    # EMS
    "ems":       ["express mail","тезкор","ekspress pochta","xalqaro tezkor","international express"],

    # Bir qadam
    "bir qadam": ["1 kunda","bir kunda","tez","tezkor","1 ish kuni","bir ish kuni"],
    "birqadam":  ["bir qadam","1 kunda","tezkor","viloyat","regional"],

    # Fulfillment
    "fulfilment":["ombor","warehouse","logistika","internet-do'kon","onlayn savdo","saqlash","yetkazish"],
    "ombor":     ["fulfilment","warehouse","saqlash","складирование"],

    # Sekogramma
    "sekogramma":["brayl","ko'zi ojiz","cecogram","слепые","бесплатно","free"],
    "brayl":     ["sekogramma","ko'zi ojiz","незрячий","шрифт"],

    # Xalqaro
    "xalqaro":   ["international","chet el","abroad","зарубеж","chetga","overseas","global"],
    "chet":      ["xalqaro","international","abroad","зарубеж","chetga"],
    "rossiya":   ["россия","russia","rf","москва","moscow"],
    "qozogiston":["казахстан","kazakhstan","almaty","астана"],
    "turkiya":   ["турция","turkey","istanbul","анкара"],
    "germaniya": ["германия","germany","berlin","deutschland"],
    "aqsh":      ["сша","usa","america","new york","states"],
    "xitoy":     ["китай","china","beijing","zhongquo"],

    # Og'irlik
    "kg":        ["kilogram","кг","kilogramm","vazn","og'irlik","weight"],
    "vazn":      ["kg","og'irlik","weight","кг","gramm","g"],
    "og'irlik":  ["vazn","kg","weight","кг"],

    # MDH
    "mdh":       ["sng","снг","cis","qo'shni mamlakatlar","rossiya","qozog'iston","belarus"],

    # Qo'shimcha xizmatlar
    "sugurta":   ["sug'urta","elon qilingan qiymat","страховка","insurance","kompensatsiya","declaratsiya"],
    "kompensatsiya":["sug'urta","yo'qolsa","isrobovlasa","возмещение","compensation"],

    # Manzil
    "manzil":    ["adres","address","адрес","uy","ko'cha","viloyat","shahar"],
    "indeks":    ["pochta indeksi","zip","postal code","индекс"],

    # Shikoyat
    "shikoyat":  ["muammo","ariza","жалоба","complaint","murojaat","yordam"],
    "muammo":    ["shikoyat","xato","problem","issue","жалоба","kelmadi","yo'qoldi"],

    # Ish vaqti
    "ish vaqti": ["soat","qachon","kun","schedule","работает","ishlaydi","nechada"],
    "shanba":    ["суббота","saturday","dam olish","weekend"],
    "yakshanba": ["воскресенье","sunday","dam olish","yopiq","closed"],
}


# ── Kategoriya kalitlari (boyitilgan) ─────────────────────────────────────────

_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "bir_qadam": [
        # O'zbekcha
        "bir qadam","birqadam","bir-qadam","1 kunda yetkazish","1 kunda",
        "bir ish kunida","1 ish kuni","tez viloyat","viloyatga tez",
        "toshkentdan viloyatga 1 kunda","viloyatdan toshkentga 1 kunda",
        "samarqandga 1 kunda","nukusga 1 kunda","termizga 1 kunda",
        "urganchga 1 kunda","buxoroga 1 kunda","farg'onaga 1 kunda",
        "andijonga 1 kunda","namanganga 1 kunda","qarshiga 1 kunda",
        "navoiyga 1 kunda","jizzaxga 1 kunda",
        "ekspress ichki","tez ichki yetkazish","arzon tez pochta",
        "bir kunda boradi","bir kunda yetadi","ertaga yetadi",
        # Ruscha
        "один шаг","одним шагом","доставка за 1 день","за один день",
        "экспресс внутри","срочно в регион","ташкент регион 1 день",
        # Inglizcha
        "one step","1 day delivery","same day region","next day delivery uzbekistan",
        "express domestic","fast regional","tashkent region express",
    ],

    "hybrid_mail": [
        # O'zbekcha
        "gibrid pochta","gibrit pochta","gibrid","gibrit",
        "gibrid narx","gibrit narx","gibrid xizmat",
        "elektron hujjat qog'ozda","elektron yuborib qog'ozda olish",
        "onlayn yuborish qog'ozda yetkazish","chop etib yetkazish",
        "ofisdan chiqmasdan xat","rasmiy xat onlayn",
        "sudga xabarnoma onlayn","qarzdorga xat onlayn",
        "pretenziya onlayn yuborish","talabnoma onlayn pochta",
        "hujjatni email orqali pochta","email yuborib pochta",
        "yuridik xat onlayn","soliq xabarnomasi pochta",
        "shartnoma bekor qilish xati","ogohlantirish xati onlayn",
        "bank xabarnomasi pochta","jarima xabarnomasi",
        "sud xabarnomasini onlayn topshirish",
        # Ruscha
        "гибридная почта","гибридная","гибрид","электронный документ бумагой",
        "досудебная претензия онлайн","уведомление онлайн почтой",
        "расторжение договора письмо онлайн","налоговое уведомление почта",
        "юридически значимое письмо онлайн","не выходя из офиса письмо",
        # Inglizcha
        "hybrid mail","electronic to paper","digital to postal","online letter physical",
        "pre-trial claim postal","legal notice online","court notice mail",
        "official letter without office","registered letter online",
    ],

    "courier": [
        # O'zbekcha
        "kuryerlik xizmati","kuryer xizmati","kuryer chaqirish",
        "kurierlik xizmati","kurierlik","kurier","kuryerlik",
        "uyga keladi","uyga yetkazish","uydan olish","uydan jo'natish",
        "eshikma eshik","kapiga keladi","kaptiga keladi",
        "kuryer chaqirmoqchiman","kuryer buyurtma","kuryer narxi",
        "uyimga kelsinmi","uydan olib ketish","uyga olib kelish",
        "kuryer bilan yuborish","kuryer bilan yetkazish",
        "door to door","kuryer haq","chaqirish haqi",
        # Ruscha
        "курьерская служба","вызов курьера","курьер на дом",
        "курьер заберёт","доставка на дом","привезут домой",
        "от двери до двери","курьерская доставка","вызвать курьера",
        "курьер приедет","забрать отправление дома",
        # Inglizcha
        "courier service","door to door","home delivery","pickup from home",
        "courier pickup","courier delivery","send from home",
    ],

    "cecogram": [
        # O'zbekcha
        "sekogramma","ko'zi ojiz","brayl","bepul pochta ojizlar uchun",
        "ko'rish qobiliyati yo'q","nogironlar uchun pochta",
        "brayl alifbosi","brayl kitob","audio kitob pochta",
        "ko'zi ojizlar kutubxonasi","nogironlar muassasasi pochta",
        "7 kg bepul","brayl materiallar","maxsus pochta ojizlar",
        # Ruscha
        "секограмма","незрячих","брайль","бесплатная пересылка слепым",
        "литература для незрячих","книги брайль почта",
        "аудиокниги для слепых","бесплатная почта инвалидам",
        # Inglizcha
        "cecogram","braille mail","blind persons post","free postal blind",
        "braille books mail","audio books post blind",
    ],

    "ems": [
        # O'zbekcha
        "ems","express mail service","tezkor xalqaro","xalqaro tezkor",
        "chet elga tezkor","xalqaro ekspress","xalqaro 1-5 kun",
        "xalqaro jo'natma tez","chet elga jo'natma tez",
        "germaniyaga tez","rossiyaga tez","aqshga tez","turkiyaga tez",
        "xitoyga tez","baaга tez","yaponiyaga tez",
        "xalqaro pochta ekspress","tezkor chet el yetkazish",
        "ekspress xalqaro yetkazish","xalqaro trek kod",
        "ems zona","ems tarif","ems narxi","ems necha kun",
        "ems bilan yuborish","ems xizmati",
        # Ruscha
        "экспресс почта","международный экспресс","эмс","ems","срочная международная",
        "быстрая доставка за рубеж","экспресс в россию","экспресс в германию",
        "международная экспресс-почта","трекинг международный",
        # Inglizcha
        "ems","express mail service","international express","fast international",
        "express to russia","express to germany","international tracked",
        "ems zones","ems rates","fastest international post",
    ],

    "small_packet": [
        # O'zbekcha
        "mayda paket","kichik paket","mayda jo'natma","kichik buyum pochta",
        "2 kg gacha buyum","kiyim yuborish","aksessuar yuborish",
        "sovg'a yuborish","kosmetika yuborish","kichik tovar pochta",
        "mayda paket narxi","kichik buyum chet elga",
        "mayda paket xalqaro","kichik paket ichki",
        # Ruscha
        "мелкий пакет","небольшой пакет","маленький пакет",
        "мелкие предметы до 2 кг","отправить одежду","небольшой подарок",
        "мелкий пакет за границу","мелкий пакет по узбекистану",
        # Inglizcha
        "small packet","small parcel","tiny parcel","light packet",
        "send clothing","send accessories","send gift under 2kg",
        "small packet international","light item post",
    ],

    "wrapper": [
        # O'zbekcha
        "banderol","kitob yuborish","kitob pochta","jurnal yuborish",
        "gazeta yuborish","bosma nashr yuborish","qo'lyozma yuborish",
        "fotosurat yuborish pochta","dissertatsiya yuborish",
        "bosma material pochta","banderol narxi","banderol xizmati",
        "banderol nima","kitob jo'natish","jurnal jo'natish",
        # Ruscha
        "бандероль","отправить книгу","книга по почте","журнал по почте",
        "печатная продукция почта","рукопись по почте","фото почтой",
        "бандероль стоимость","отправить литературу",
        # Inglizcha
        "banderol","wrapper","book mail","send books","magazine post",
        "printed matter mail","manuscript post","newspaper mail",
    ],

    "letter": [
        # O'zbekcha
        "xat yuborish","buyurtmali xat","oddiy xat","xat narxi",
        "xat jo'natish","hujjat xat bilan","konvertda yuborish",
        "muhim hujjat yuborish","rasmiy xat","xat bilan shartnoma",
        "pretenziya xati","talabnoma xati","sud uchun xat",
        "xat nima","xat xizmati","xat trek","xat kuzatish",
        "2 kg gacha xat","buyurtmali xat narxi","oddiy xat narxi",
        # Ruscha
        "письмо","заказное письмо","простое письмо","отправить документ",
        "ценное письмо","письмо с трекингом","письмо по почте",
        "конверт с документами","письмо официальное",
        # Inglizcha
        "letter","registered letter","ordinary letter","send document by mail",
        "letter service","insured letter","official letter post",
    ],

    "postcard": [
        # O'zbekcha
        "otkritka","pochta varaqchasi","tabriknoma pochta",
        "tabrik kartochkasi","salomlashish otkritka",
        "otkritka yuborish","otkritka narxi","otkritka xizmati",
        "otkritka nima","otkritka jo'natish","ochiq xat",
        "tabrik yuborish pochta","bayram otkritka",
        # Ruscha
        "открытка","почтовая карточка","поздравление почтой",
        "открытка стоимость","открытка куда","открытка по почте",
        # Inglizcha
        "postcard","greeting card mail","postal card","send postcard",
        "postcard service","postcard price","open card post",
    ],

    "parcel": [
        # O'zbekcha
        "posilka yuborish","posilka narxi","posilka xizmati",
        "posilka nima","tovar yuborish","yuk yuborish",
        "2 kg dan ortiq","og'ir jo'natma","katta jo'natma",
        "posilka qancha","posilka trek","posilka kuzatish",
        "posilka qabul","posilka olish","oziq-ovqat yuborish pochta",
        "elektronika yuborish pochta","posilka ichki","posilka xalqaro",
        "posilka saqlash","posilka yer usti","posilka havo",
        "cn23","bojxona posilka","posilka deklaratsiya",
        # Ruscha
        "посылка","отправить посылку","посылка цена","посылка трекинг",
        "посылка хранение","посылка вес","посылка получить",
        "товары почтой","посылка внутри узбекистана","посылка за границу",
        # Inglizcha
        "parcel","send parcel","parcel price","parcel tracking",
        "send goods","parcel service","parcel domestic","parcel international",
        "parcel storage","cn23 declaration",
    ],

    "money_transfer": [
        # O'zbekcha
        "pul o'tkazma","pul yuborish","pul otkazma","pul jo'natish",
        "pul olish pochta","komissiya pul","pul transfer",
        "elektron pul o'tkazma","pochta orqali pul","pasport bilan pul",
        "3 foiz komissiya","6 foiz komissiya","aliment yuborish",
        "nafaqa yuborish pochta","bank kartasisiz pul",
        "pul qancha vaqtda yetadi","pul olish bo'lim",
        # Ruscha
        "денежный перевод","пул юборish","деньги через почту",
        "перевод комиссия","3%","6%","алименты почта",
        "пособие почта","перевод без карты","деньги получить на почте",
        # Inglizcha
        "money transfer","postal transfer","send money post","cash transfer",
        "money order","3 percent fee","6 percent fee","alimony post",
    ],

    "fulfilment": [
        # O'zbekcha
        "fulfillment","fulfilment","ombor xizmati","logistika xizmati",
        "internet-do'kon ombor","onlayn savdo logistika",
        "tovar saqlash pochta","buyurtma ishlov","qadoqlash xizmati",
        "markirovka xizmati","otsifrovka","telegram shop ombor",
        "instagram shop logistika","e-commerce ombor",
        "tovar qabul saqlash yetkazish","fulfillment narxi",
        "ombor kunlik haq","pallet saqlash","qaytarilgan tovar",
        # Ruscha
        "фулфилмент","склад для магазина","логистика интернет-магазин",
        "хранение товаров почта","обработка заказов","маркировка товаров",
        "оцифровка товаров","склад и доставка","фулфилмент цена",
        # Inglizcha
        "fulfilment","fulfillment","warehouse service","ecommerce logistics",
        "online store warehouse","order processing","labeling service",
        "product photography","return handling","pallet storage",
    ],

    "storage": [
        # O'zbekcha
        "saqlash muddati","bildirishnoma","ikkinchi bildirishnoma",
        "saqlash to'lovi","kunlik to'lov saqlash","necha kun bepul",
        "bepul saqlash","posilka necha kun turadi","kech olsam qancha",
        "saqlash narxi","posilka bo'limda turadi","ish kuni saqlash",
        "shanba hisoblanadimi saqlash","olib ketgan kun",
        "bir oy saqlash","qaytariladi olinmasa","saqlash 3500",
        # Ruscha
        "срок хранения","второе извещение","плата за хранение",
        "бесплатное хранение","сколько дней хранится","хранение рабочих дней",
        "хранение выходные","оплата хранения","получить посылку поздно",
        # Inglizcha
        "storage period","second notice","storage fee","free storage days",
        "how long stored","storage business days","late pickup fee",
        "parcel storage notice","storage per day",
    ],

    "m_bag": [
        # O'zbekcha
        "m qopi","m-qop","m qop","m xaltasi",
        "ko'p kitob yuborish","ommaviy bosma nashr","jurnallar ommaviy",
        "nashriyot pochta","kitoblar ommaviy","ko'p nashr jo'natish",
        "m qop narxi","m qop nima","m bag xizmati",
        # Ruscha
        "м мешок","м-мешок","м мешок почта","ем мешок",
        "массовая рассылка книг","пачка журналов почта",
        "издательство почта","много книг почтой","м мешок цена",
        # Inglizcha
        "m bag","m-bag","bulk book mailing","mass printed matter",
        "publisher mail","bulk magazine post","m bag price","m bag service",
    ],

    "postcard_cis": [
        "otkritka mdh","otkritka sng","rossiyaga otkritka",
        "qozog'istonga otkritka","belarusga otkritka","ukrainaga otkritka",
        "mdh otkritka narxi","sng otkritka","mdh tabriknoma",
        "открытка снг","открытка в россию","открытка в казахстан",
        "postcard cis","postcard to russia","postcard to kazakhstan",
        "cis postcard price",
    ],

    "letter_cis": [
        "xat mdh","xat sng","rossiyaga xat","qozog'istonga xat",
        "belarusga xat","ukrainaga xat","mdh xat narxi","sng xat",
        "xalqaro xat sng","письмо снг","письмо в россию",
        "письмо в казахстан","письмо в беларусь","международное письмо снг",
        "letter cis","letter to russia","letter to kazakhstan","cis letter price",
    ],

    "ems_far_abroad": [
        "ems xalqaro","ems germaniya","ems aqsh","ems turkiya",
        "ems yaponiya","ems avstraliya","ems kanada","ems zona 5","ems zona 6",
    ],
}


# ── Category → Service ID mapping (boyitilmadi, to'liq) ─────────────────────

_CATEGORY_TO_SERVICE: dict[str, str] = {
    "bir_qadam":               "service_bir_qadam_001",
    "hybrid_mail":             "service_hybrid_mail_001",
    "courier":                 "service_courier_001",
    "cecogram":                "service_cecogram_001",
    "ems":                     "service_ems_001",
    "small_packet":            "service_small_packet_001",
    "wrapper":                 "service_wrapper_001",
    "letter":                  "service_letter_001",
    "postcard":                "service_postcard_001",
    "parcel":                  "service_parcel_001",
    "money_transfer":          "service_electronic_money_transfer_001",
    "fulfilment":              "service_fulfilment_001",
    "storage":                 "service_parcel_wrapper_storage_001",
    "m_bag":                   "service_m_bag_001",
    "postcard_cis":            "service_postcard_cis_001",
    "letter_cis":              "service_letter_cis_001",
    "additional_services":     "service_additional_services_001",
    "additional_international":"service_additional_international_001",
    "email_message_print":     "service_email_message_print_001",
    "international_guide":     "service_international_guide_001",
    "management_contact":      "service_management_contact_001",
    "general_guide":           "general_uzpost_guide_001",
}


# ── Comparison pairs (boyitilgan) ─────────────────────────────────────────────

_COMPARISON_PAIRS: dict[str, list[str]] = {
    "bir_qadam":    ["service_bir_qadam_001",   "service_courier_001"],
    "courier":      ["service_courier_001",     "service_bir_qadam_001"],
    "ems":          ["service_ems_001",         "service_small_packet_001"],
    "parcel":       ["service_parcel_001",      "service_small_packet_001"],
    "letter":       ["service_letter_001",      "service_wrapper_001"],
    "small_packet": ["service_small_packet_001","service_parcel_001"],
    "wrapper":      ["service_wrapper_001",     "service_letter_001"],
    "postcard":     ["service_postcard_001",    "service_letter_001"],
    "hybrid_mail":  ["service_hybrid_mail_001", "service_letter_001"],
    "m_bag":        ["service_m_bag_001",       "service_wrapper_001"],
    "fulfilment":   ["service_fulfilment_001",  "service_parcel_001"],
}


# ── Intent default services (boyitilgan) ──────────────────────────────────────

_INTENT_DEFAULT_SERVICES: dict[str, list[str]] = {
    "tracking":   ["service_ems_001", "service_bir_qadam_001", "service_parcel_001"],
    "price":      ["service_bir_qadam_001", "service_courier_001", "service_parcel_001"],
    "storage":    ["service_parcel_wrapper_storage_001", "service_parcel_001"],
    "complaint":  ["service_ems_001", "service_bir_qadam_001", "service_parcel_001"],
    "coverage":   ["service_bir_qadam_001", "service_courier_001", "service_ems_001"],
    "comparison": ["service_bir_qadam_001", "service_courier_001"],
    "info":       ["general_uzpost_guide_001"],
    "restriction":["service_parcel_001", "service_ems_001", "service_small_packet_001"],
    "international": ["service_ems_001", "service_international_guide_001", "service_parcel_001"],
    "contact":    ["service_management_contact_001"],
    "money":      ["service_electronic_money_transfer_001"],
    "fulfilment": ["service_fulfilment_001"],
}


# ── Category exclusions (boyitilgan) ──────────────────────────────────────────

_CATEGORY_EXCLUSIONS: dict[str, set[str]] = {
    "bir_qadam":    {"courier","ems","parcel","small_packet","m_bag",
                     "postcard","cecogram","letter_cis","postcard_cis"},
    "courier":      {"bir_qadam","ems","m_bag","cecogram",
                     "postcard_cis","letter_cis"},
    "hybrid_mail":  {"courier","bir_qadam","parcel","ems","m_bag",
                     "small_packet","cecogram"},
    "ems":          {"parcel","small_packet","bir_qadam","courier",
                     "m_bag","cecogram","storage","fulfilment"},
    "small_packet": {"parcel","ems","bir_qadam","m_bag","fulfilment",
                     "storage","cecogram"},
    "wrapper":      {"parcel","letter","ems","bir_qadam","courier",
                     "fulfilment","money_transfer"},
    "letter":       {"wrapper","parcel","ems","bir_qadam","courier",
                     "fulfilment","money_transfer","m_bag"},
    "postcard":     {"letter","parcel","ems","bir_qadam","courier",
                     "money_transfer","fulfilment","m_bag"},
    "fulfilment":   {"parcel","storage","bir_qadam","courier","ems",
                     "small_packet","letter","postcard","wrapper"},
    "storage":      {"fulfilment","bir_qadam","courier","ems",
                     "letter","postcard","wrapper","small_packet"},
    "money_transfer":{"parcel","letter","ems","bir_qadam","courier",
                      "small_packet","wrapper","postcard","fulfilment"},
    "cecogram":     {"wrapper","parcel","letter","ems","bir_qadam",
                     "courier","small_packet","money_transfer","fulfilment"},
    "m_bag":        {"parcel","wrapper","ems","bir_qadam","courier",
                     "letter","postcard","small_packet","fulfilment"},
    "postcard_cis": {"postcard","letter","parcel","ems","bir_qadam",
                     "courier","m_bag","fulfilment","money_transfer"},
    "letter_cis":   {"letter","postcard","parcel","ems","bir_qadam",
                     "courier","m_bag","fulfilment","money_transfer"},
    "hybrid_mail":  {"courier","bir_qadam","parcel","ems","m_bag",
                     "small_packet","cecogram"},
}


# ── Intent patterns (boyitilgan) ──────────────────────────────────────────────

_INTENT_PATTERNS: dict[str, list[str]] = {
    "price": [
        # O'zbekcha
        "narx","qancha","tarif","pul","so'm","turadi","tolov","tanishtir",
        "ko'rsat","narxlarini","tariflarni","hisoblash","qancha chiqadi",
        "qancha oladi","qancha to'layman","formula","haq","summa","arzon",
        "qimmat","chegirma","bepulmi","pulli","to'lovmi","narx bor",
        "qancha narx","misol hisoblash","hisob","hisoblab",
        # Ruscha
        "сколько","цена","стоит","тариф","расчёт","стоимость",
        "дешевле","дороже","бесплатно","платно","комиссия","сумма",
        # Inglizcha
        "price","cost","fee","how much","rate","charge","tariff",
        "calculate","cheaper","expensive","free","paid","commission",
    ],

    "tracking": [
        # O'zbekcha
        "trek","kuzat","qayerda","yetib keldi","kelmadi","status","holat",
        "jo'natma haqida","qayerda turibdi","yetib borganmi","bordi",
        "kelganmi","kelmadimi","qachon keladi","necha kunda yetadi",
        "trek raqam","kuzatish raqami","bordi keldi","qayerdaligi",
        # Ruscha
        "трек","отследить","где посылка","статус","отправление дошло",
        "когда придёт","трек-номер","отслеживание","где моё","пришло",
        # Inglizcha
        "track","where is","status","arrived","tracking number",
        "shipment status","locate","where my","delivery status",
    ],

    "storage": [
        # O'zbekcha
        "saqlash","saqlanadi","muddat","necha kun","bildirishnoma",
        "ikkinchi bildirishnoma","saqlash to'lovi","bepul saqlash",
        "kech olsam","qancha to'layman saqlash","ish kuni saqlash",
        "shanba hisoblanadimi","olib ketgan kun","bir oy","qaytariladi",
        "3500","kunlik to'lov","posilka turadi",
        # Ruscha
        "хранение","хранится","второе извещение","срок хранения",
        "плата хранение","сколько дней хранится","забрать","оплата хранение",
        # Inglizcha
        "storage","stored","second notice","storage fee","free storage",
        "how long","pickup deadline","storage days","storage charge",
    ],

    "coverage": [
        # O'zbekcha
        "qayerlarga","boradi","yetkazadi","yo'nalish","qaysi shaharlar",
        "qaysi viloyatlar","qaysi mamlakatlar","qamrov","xizmat hududи",
        "ishlaydi","qayerga ishlaydi","qayerga boradi","mamlakat ro'yxati",
        "shaharlar ro'yxati","viloyat markazlari","marshrut","yo'nalishlar",
        # Ruscha
        "куда","доставляет","маршрут","направления","какие города",
        "какие страны","зона покрытия","регионы","куда работает",
        # Inglizcha
        "where deliver","destinations","cities","countries","coverage",
        "service area","regions","routes","which cities","which countries",
    ],

    "restriction": [
        # O'zbekcha
        "mumkinmi","taqiq","cheklov","og'irlik","maximum","minimal",
        "ruxsat","yuborsa bo'ladimi","yuborib bo'ladimi","man etilgan",
        "taqiqlangan","nima yuborsa bo'lmaydi","qanday buyum",
        "limit","chek","chegarа","qoida","shart","talаb",
        "portlovchi","narkotik","qurol","spirtli","dori","o'simlik",
        # Ruscha
        "можно ли","запрещено","ограничение","максимальный","минимальный",
        "что нельзя","разрешено","запрет","лимит","правило","требование",
        # Inglizcha
        "allowed","restriction","limit","prohibited","maximum","minimum",
        "can i send","forbidden","rules","requirements","what not to send",
    ],

    "comparison": [
        # O'zbekcha
        "farqi","yaxshi","taqqosla","vs","qaysi yaxshi","qaysi tez",
        "qaysi arzon","qaysi ishonchli","farq nimada","qaysi biri",
        "yaxshiroq","aniqroq","solishtirsam","bir-biriga nisbatan",
        "bir qadam yoki kuryer","ems yoki posilka","banderol yoki xat",
        # Ruscha
        "разница","лучше","сравни","отличие","vs","что выбрать",
        "быстрее","дешевле","надёжнее","какой лучше","что хуже",
        # Inglizcha
        "compare","difference","better","vs","which is","faster",
        "cheaper","more reliable","or","versus","choose between",
    ],

    "complaint": [
        # O'zbekcha
        "kelmadi","yo'qoldi","kechikdi","muammo","shikoyat","buzilgan",
        "shikastlangan","noto'g'ri yetkazildi","noto'g'ri manzil",
        "qaytib kelmadi","topilmadi","yo'q","yo'qolgan","xato",
        "rad etilgan","qabul qilinmagan","past sifat","xodim muammosi",
        "1165 javob bermaydi","pochta javob bermaydi",
        # Ruscha
        "не пришло","потерялось","задержка","проблема","жалоба",
        "повреждено","не то пришло","неправильный адрес","пропало",
        "ошибка","испорчено","недовольство","плохой сервис",
        # Inglizcha
        "not arrived","lost","delayed","problem","complaint","damaged",
        "wrong address","missing","issue","poor service","broken",
    ],

    "international": [
        # O'zbekcha
        "xalqaro","chet el","chet elga","chetga","xorijga","mamlakatga",
        "rossiyaga","germaniyaga","aqshga","turkiyaga","xitoyga",
        "chet el tariflari","xalqaro yetkazish","xalqaro jo'natma",
        "bojxona","deklaratsiya","cn23","sug'urta xalqaro",
        "mdh mamlakatlar","uzoq xorij","havo pochta","yer usti xalqaro",
        # Ruscha
        "международный","за рубеж","заграница","в россию","в германию",
        "таможня","декларация","международная доставка","зарубежная почта",
        # Inglizcha
        "international","abroad","overseas","foreign","to russia","to germany",
        "customs","declaration","cn23","international shipping","global",
    ],

    "contact": [
        # O'zbekcha
        "telefon","aloqa","murojaat","bosh direktor","rahbariyat",
        "1165","info@pochta.uz","pochta telefon","qabul vaqti",
        "shikoyat qayerga","bog'lanish","pochta manzili","qo'ng'iroq",
        "pochta email","pochta sayt","uz.post","yordam","support",
        # Ruscha
        "телефон","связь","контакт","директор","руководство",
        "жалоба куда","позвонить","адрес почты","приём",
        # Inglizcha
        "phone","contact","call","director","management",
        "complaint","support","address","reception","1165",
    ],

    "money": [
        # O'zbekcha
        "pul","o'tkazma","transfer","komissiya","pul yuborish",
        "pul olish","aliment","nafaqa","pasport bilan pul",
        "3 foiz","6 foiz","bank kartasisiz","tezda pul",
        # Ruscha
        "деньги","перевод","комиссия","алименты","пособие",
        "без карты","получить деньги","отправить деньги",
        # Inglizcha
        "money","transfer","send money","commission","alimony",
        "without card","receive money","3 percent","6 percent",
    ],
}


# ── Intent → Chunk intents (boyitilgan) ───────────────────────────────────────

_INTENT_TO_CHUNK_INTENTS: dict[str, list[str]] = {
    "price": [
        "pricing","callout_fee","pricing_within_region",
        "pricing_between_regions","calculation_formula","definition",
        "pricing_domestic","pricing_cis","pricing_far_abroad",
        "pricing_international","rounding_rules","tariff",
        "declared_value_fee","overview",
    ],
    "storage": [
        "storage","storage_period","restriction","notices",
        "return_policy","examples","multiple_items",
        "storage_core_001","storage_return_001",
    ],
    "tracking": [
        "tracking","support_contact","definition",
        "storage_tracking","tracking_problems",
        "tracking_and_problems",
    ],
    "coverage": [
        "coverage","service_area","restriction","definition",
        "definition_coverage","delivery_time_coverage",
        "what_can_send","weight_limit",
    ],
    "restriction": [
        "restriction","weight_limit","coverage","what_can_send",
        "customs_services","customs","packaging_rules",
        "labeling","what_cannot_send",
    ],
    "comparison": [
        "comparison","vs_bir_qadam","vs_parcel","definition","pricing",
        "comparison_international","vs_courier","vs_ems",
    ],
    "complaint": [
        "problem_and_complaint","tracking","support_contact",
        "problem_and_complaint","search","complaint",
    ],
    "international": [
        "definition","pricing","delivery_time","coverage",
        "customs_services","declared_value","notification_services",
        "rounding_rules","delivery_tracking_insurance",
        "pricing_cis","pricing_far_abroad","which_service_to_choose",
        "regions_customs","comparison",
    ],
    "contact": [
        "general_contact","leadership_contacts","online_resources",
        "support_contact","practical",
    ],
    "money": [
        "pricing","calculation_examples","preferential_case",
        "limits","problems_and_complaint","how_to_send","how_to_receive",
    ],
    "fulfilment": [
        "definition","labeling_digitization","storage","order_processing",
        "returns_start","how_to_start","pricing_processing",
    ],
    "info": [
        "definition","pricing","how_it_works","coverage","eligibility",
        "delivery_time","definition_types","which_service_to_choose",
        "service_selection","weight_rules","storage_complaint",
        "international_money","packaging_heavy_cost_mistakes",
    ],
}


# ── LOW PRIORITY categories ───────────────────────────────────────────────────

_LOW_PRIORITY_CATEGORIES = {
    "general_guide","international_guide","management_and_contact",
    "management_contact","email_printing","additional_services_international",
}