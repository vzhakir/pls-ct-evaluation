"""
evaluation/test_cases.py
─────────────────────────
75 test case REALISTIS untuk RAG evaluation suite.

Versi ini dipangkas dari 110 test case menjadi 75 test case pertama
(eval-001 sampai eval-075), sesuai kebutuhan evaluasi yang lebih ringan.

Schema setiap test case:
- query
- relevant_keywords
- cognitive
- session_id
- query_type
- context_note

TIDAK ADA reference_answer — evaluasi answer quality menggunakan chunk GT
yang di-retrieve oleh RAG sebagai referensi/proxy.
"""

from typing import Dict, List

TEST_CASES: List[Dict] = [{'query': 'Saya sudah baca definisi Computational Thinking dari berbagai sumber dan semuanya '
           "bilang CT itu tentang 'formulasi masalah dan solusi'. Tapi berpikir logis yang saya "
           'pelajari di matematika SMA juga tentang formulasi masalah. Jadi bedanya apa CT dengan '
           'berpikir logis biasa?',
  'relevant_keywords': ['computational thinking',
                        'formulasi masalah',
                        'berpikir logis',
                        'algoritme',
                        'abstraksi'],
  'cognitive': '2PAR',
  'session_id': 'eval-001',
  'query_type': 'gap',
  'context_note': 'Mahasiswa tidak bisa membedakan CT dengan berpikir logis SMA'},
 {'query': 'Di catatan saya tulis AADP = Abstraksi → Algoritme → Dekomposisi → Pattern Recognition '
           'dan saya kerjakan soal dengan urutan itu. Tapi nilai saya jelek. Kata teman, AADP '
           'bukan urutan. Lalu AADP itu apa sebenarnya?',
  'relevant_keywords': ['AADP', 'pilar', 'dekomposisi', 'urutan langkah', 'pseudocode'],
  'cognitive': '1TAR',
  'session_id': 'eval-002',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa salah mengira AADP adalah urutan prosedural'},
 {'query': 'Dosen bilang kita hidup di era VUCA dan CT itu penting untuk karir. Tapi saya mau jadi '
           'dokter, bukan programmer. Apakah CT relevan untuk dokter di era VUCA, atau ini hanya '
           'penting untuk orang teknik saja?',
  'relevant_keywords': ['VUCA', 'computational thinking', 'karir', 'dokter', 'dekomposisi'],
  'cognitive': '3TGI',
  'session_id': 'eval-003',
  'query_type': 'out_of_scope',
  'context_note': 'Mahasiswa tidak berlatar teknik mempertanyakan relevansi CT untuk karir '
                  'non-teknik'},
 {'query': 'Di silabus ada materi CT dan materi ICT Literacy. Saya bingung — apakah ICT Literacy '
           'itu bagian dari CT atau CT bagian dari ICT Literacy? Atau keduanya hal yang sama '
           'sekali berbeda?',
  'relevant_keywords': ['ICT literacy',
                        'computational thinking',
                        'literasi digital',
                        'AADP',
                        'teknologi'],
  'cognitive': '2TAI',
  'session_id': 'eval-004',
  'query_type': 'cross_topic',
  'context_note': 'Mahasiswa bingung hubungan hierarki CT dan ICT Literacy'},
 {'query': 'Teman saya bilang saya punya digital footprint besar karena sering pakai Instagram. '
           'Tapi saya hampir tidak pernah posting — saya hanya scroll dan nonton video orang lain. '
           'Apa betul saya tetap punya digital footprint meski tidak pernah upload apapun?',
  'relevant_keywords': ['digital footprint',
                        'jejak digital',
                        'media sosial',
                        'privasi',
                        'algoritme'],
  'cognitive': '3PAR',
  'session_id': 'eval-005',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa kira digital footprint hanya dari konten yang diunggah'},
 {'query': "Di slide ada dua istilah: 'ICT Literacy' dan 'Literacy with ICT'. Keduanya terlihat "
           'sama — intinya kan paham teknologi. Kenapa perlu dua istilah berbeda untuk hal yang '
           'sama?',
  'relevant_keywords': ['ICT literacy', 'LwICT', 'literasi', 'digital', 'computational thinking'],
  'cognitive': '2PAI',
  'session_id': 'eval-006',
  'query_type': 'gap',
  'context_note': 'Mahasiswa tidak paham perbedaan antara ICT Literacy dan LwICT'},
 {'query': 'Saya buat aplikasi untuk tugas dan kumpulkan data nama dan email teman. Dari materi '
           'etika digital, apakah saya perlu minta izin dulu sebelum menyimpan data mereka? Dan '
           'kalau sudah terlanjur, apa yang harus saya lakukan?',
  'relevant_keywords': ['etika digital', 'privasi', 'data pribadi', 'izin', 'dekomposisi'],
  'cognitive': '4TAI',
  'session_id': 'eval-007',
  'query_type': 'out_of_scope',
  'context_note': 'Mahasiswa tanya prosedur konkret perlindungan data — detail regulasi tidak ada '
                  'di GT'},
 {'query': 'Waktu kerja kelompok, kami bagi tugas: saya bikin bagian A, teman bikin bagian B. Kata '
           'saya itu dekomposisi, tapi teman saya bilang itu bukan dekomposisi CT. Lalu apa '
           "bedanya 'bagi tugas' biasa dengan dekomposisi dalam CT?",
  'relevant_keywords': ['dekomposisi', 'sub-masalah', 'modular', 'pembagian tugas', 'abstraksi'],
  'cognitive': '1PAI',
  'session_id': 'eval-008',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa kira bagi tugas kelompok = dekomposisi CT'},
 {'query': 'Ada soal: 2 lift kapasitas masing-masing 30 kg. Ada 9 berang dengan berat: A=2, B=3, '
           'C=5, D=8, E=9, F=9, G=12, H=12, I=22 kg. Bagaimana dekomposisi masalah ini untuk '
           'memaksimalkan jumlah berang yang terangkut?',
  'relevant_keywords': ['dekomposisi', 'optimasi', 'lift', 'sub-masalah', 'rekursi', 'greedy'],
  'cognitive': '2PAR',
  'session_id': 'eval-009',
  'query_type': 'application',
  'context_note': 'Soal lift berang-berang ada di GT_SUBTOPIK_01 — apakah RAG retrieve dokumen '
                  'yang tepat'},
 {'query': 'Waktu belajar fungsi, dosen bilang setiap fungsi seharusnya hanya melakukan satu hal. '
           'Itu berhubungan dengan dekomposisi kan? Tapi saya bingung — apakah fungsi dalam '
           'pemrograman itu implementasi dari dekomposisi CT?',
  'relevant_keywords': ['fungsi', 'dekomposisi', 'modularitas', 'DRY', 'abstraksi', 'sub-masalah'],
  'cognitive': '3TAI',
  'session_id': 'eval-010',
  'query_type': 'cross_topic',
  'context_note': 'Mahasiswa menghubungkan konsep fungsi dengan dekomposisi CT'},
 {'query': 'Kalau saya dekomposisi masalah, sampai seberapa dalam saya harus memecahnya? Misalnya '
           "'bikin kue' — apakah saya pecah sampai level 'gerakkan jari untuk ngaduk' atau cukup "
           "sampai 'campurkan bahan'? Apa ada kriteria kapan dekomposisi sudah cukup?",
  'relevant_keywords': ['dekomposisi', 'sub-masalah', 'granularitas', 'abstraksi', 'goal'],
  'cognitive': '4PGR',
  'session_id': 'eval-011',
  'query_type': 'gap',
  'context_note': 'Mahasiswa tidak tahu stopping criterion untuk dekomposisi'},
 {'query': 'Dari soal lift 2 buah kapasitas 30 kg dengan 9 berang (A:2, B:3, C:5, D:8, E:9, F:9, '
           'G:12, H:12, I:22), dosen minta saya lakukan abstraksi dulu sebelum cari solusi. Apa '
           'yang dimaksud abstraksi untuk soal ini — apa yang perlu difokuskan dan diabaikan?',
  'relevant_keywords': ['abstraksi', 'goal', 'batasan', 'data', 'dekomposisi', 'formulasi'],
  'cognitive': '3PAI',
  'session_id': 'eval-012',
  'query_type': 'application',
  'context_note': 'Soal abstraksi lift berang-berang ada di GT_SUBTOPIK_02'},
 {'query': 'Saya bikin model data untuk sistem nilai mahasiswa dan saya buang semua field kecuali '
           "nama dan NIM karena 'itu abstraksi'. Tapi dosen bilang abstraksi saya salah. Katanya "
           'nilai ujian, kehadiran itu juga penting. Apakah abstraksi berarti membuang detail '
           'sebanyak mungkin?',
  'relevant_keywords': ['abstraksi', 'relevan', 'detail', 'model', 'goal', 'dekomposisi'],
  'cognitive': '4TAI',
  'session_id': 'eval-013',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa kira abstraksi = buang semua detail sebanyak mungkin'},
 {'query': "Di slide abstraksi ada istilah 'helicopter view'. Saya tidak ngerti maksudnya — apa "
           'hubungannya helikopter dengan CT? Dan bagaimana cara saya menerapkan helicopter view '
           'waktu mengerjakan soal?',
  'relevant_keywords': ['helicopter view', 'abstraksi', 'gambaran besar', 'detail', 'formulasi'],
  'cognitive': '2TGI',
  'session_id': 'eval-014',
  'query_type': 'gap',
  'context_note': 'Mahasiswa tidak paham metafora helicopter view dalam konteks abstraksi CT'},
 {'query': 'Waktu analisis data nilai ujian, saya buang semua outlier dulu baru cari pola. Apakah '
           'membuang outlier itu abstraksi atau pattern recognition? Atau keduanya sekaligus? Saya '
           'bingung mana yang mana.',
  'relevant_keywords': ['abstraksi',
                        'pattern recognition',
                        'outlier',
                        'data',
                        'pola',
                        'dekomposisi'],
  'cognitive': '5TGR',
  'session_id': 'eval-015',
  'query_type': 'cross_topic',
  'context_note': 'Mahasiswa bingung apakah membuang outlier itu abstraksi atau pattern '
                  'recognition'},
 {'query': 'Soal: bebek ke-n mendapat n coklat. Total coklat 500 buah. Bebek nomor berapa yang '
           'pertama kali tidak mendapat coklat? Saya sudah coba hitung manual tapi lama sekali.',
  'relevant_keywords': ['pola bilangan',
                        'deret segitiga',
                        'pattern recognition',
                        'rumus',
                        'modulo'],
  'cognitive': '3PGR',
  'session_id': 'eval-016',
  'query_type': 'application',
  'context_note': 'Soal coklat bebek ada di GT_SUBTOPIK_03 — apakah RAG bisa retrieve'},
 {'query': 'Soal ujian: tentukan digit terakhir dari 2 pangkat 2003. Saya tidak mungkin hitung '
           '2^2003 secara langsung. Dosen bilang pakai pattern recognition. Bagaimana caranya?',
  'relevant_keywords': ['modulo', 'siklus', 'digit terakhir', 'pola', 'perpangkatan', 'deret'],
  'cognitive': '4TGR',
  'session_id': 'eval-017',
  'query_type': 'application',
  'context_note': 'Soal digit terakhir 2^n ada di GT_SUBTOPIK_04'},
 {'query': 'Saya pikir pattern recognition di CT hanya untuk data angka atau deret matematika. '
           'Tapi dosen bilang bisa juga untuk teks atau gambar. Apakah pattern recognition di CT '
           'sama dengan machine learning pattern recognition? Atau beda?',
  'relevant_keywords': ['pattern recognition', 'pola', 'data', 'abstraksi', 'machine learning'],
  'cognitive': '5PAR',
  'session_id': 'eval-018',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa bingung apakah pattern recognition CT sama dengan ML'},
 {'query': 'Saya lihat data absensi dan nemu pola: nilai turun setiap minggu ke-5. Tapi bagaimana '
           'saya tahu ini bukan kebetulan? Apa cara CT untuk memvalidasi bahwa pola yang saya '
           'temukan itu benar?',
  'relevant_keywords': ['pattern recognition',
                        'validasi',
                        'pola',
                        'data',
                        'abstraksi',
                        'dekomposisi'],
  'cognitive': '4PGI',
  'session_id': 'eval-019',
  'query_type': 'gap',
  'context_note': 'Mahasiswa tidak tahu cara validasi pola dalam CT'},
 {'query': 'Diberikan barisan: 3, 6, 12, 24, 48. Dosen minta saya identifikasi jenis deret, '
           'tuliskan rumus suku ke-n, dan hitung suku ke-10 tanpa menghitung satu per satu.',
  'relevant_keywords': ['deret geometri',
                        'rasio',
                        'pola bilangan',
                        'rumus',
                        'deret aritmatika',
                        'modulo'],
  'cognitive': '3TGR',
  'session_id': 'eval-020',
  'query_type': 'application',
  'context_note': 'Identifikasi deret geometri dari barisan angka — ada di GT_SUBTOPIK_03'},
 {'query': 'Di tugas saya bikin flowchart karena saya kira itu lebih formal dan benar dari '
           'pseudocode. Tapi dosen lebih prefer pseudocode. Apa bedanya pseudocode dengan '
           'flowchart, dan mana yang lebih tepat untuk menggambarkan algoritme?',
  'relevant_keywords': ['pseudocode',
                        'flowchart',
                        'algoritme',
                        'notasi',
                        'for loop',
                        'percabangan'],
  'cognitive': '2PAI',
  'session_id': 'eval-021',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa salah kira flowchart lebih formal dari pseudocode'},
 {'query': 'Saya diminta menulis pseudocode algoritme Euclidean untuk mencari FPB dua bilangan. '
           'Saya tahu konsep FPB tapi tidak tahu cara tulis pseudocode-nya dengan while loop. Bisa '
           'tolong tunjukkan?',
  'relevant_keywords': ['pseudocode', 'FPB', 'while', 'Euclidean', 'modulo', 'rekursi'],
  'cognitive': '3PAR',
  'session_id': 'eval-022',
  'query_type': 'application',
  'context_note': 'Pseudocode FPB Euclidean ada di GT_SUBTOPIK_05'},
 {'query': 'Saya sudah buat algoritme untuk cari nilai terbesar dari array dan bisa jalan. Tapi '
           'dosen bilang algoritme saya kurang baik meski hasilnya benar. Apa saja ciri algoritme '
           "yang dianggap 'baik' selain menghasilkan output yang benar?",
  'relevant_keywords': ['algoritme', 'ciri', 'finiteness', 'efisiensi', 'ambigu', 'feasibility'],
  'cognitive': '1TGR',
  'session_id': 'eval-023',
  'query_type': 'gap',
  'context_note': 'Mahasiswa tidak tahu ciri algoritme baik selain correctness'},
 {'query': 'Saya buat algoritme FPB Euclidean dan ingin tahu kompleksitasnya Big-O. Apakah '
           'Euclidean itu O(log n) atau O(n)? Dan kenapa bukan O(1) walaupun sering selesai cepat '
           'untuk angka kecil?',
  'relevant_keywords': ['FPB', 'Euclidean', 'Big-O', 'kompleksitas', 'O(log n)', 'algoritme'],
  'cognitive': '5TAI',
  'session_id': 'eval-024',
  'query_type': 'out_of_scope',
  'context_note': 'Kompleksitas Big-O dari Euclidean tidak eksplisit ada di GT'},
 {'query': "Di pseudocode saya selalu tulis 'int x = 5' karena terbiasa dari C++. Tapi teman saya "
           "hanya tulis 'x = 5'. Kata dosen keduanya benar. Lalu apa konvensi pseudocode CT yang "
           'benar untuk deklarasi variabel?',
  'relevant_keywords': ['pseudocode',
                        'variabel',
                        'deklarasi',
                        'tipe data',
                        'konvensi',
                        'assignment'],
  'cognitive': '2TGI',
  'session_id': 'eval-025',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa bingung konvensi deklarasi variabel di pseudocode CT vs bahasa '
                  'pemrograman'},
 {'query': 'Soal: hitung nilai ekspresi berikut: (8 + 4) / (2 * 3) - 1. Dan juga: 15 % 4 * 2 + 7 % '
           '3. Saya sudah hitung tapi tidak yakin — apakah modulo punya prioritas yang sama dengan '
           'perkalian?',
  'relevant_keywords': ['prioritas operator',
                        'modulo',
                        'ekspresi',
                        'aritmatika',
                        'kurung',
                        'pembagian'],
  'cognitive': '2PAR',
  'session_id': 'eval-026',
  'query_type': 'application',
  'context_note': 'Soal evaluasi ekspresi ada di GT_SUBTOPIK_06'},
 {'query': 'Saya sederhanakan ekspresi: NOT (A OR B) menjadi NOT A AND NOT B. Tapi teman saya '
           'bilang itu salah — harusnya NOT A OR NOT B. Padahal saya rasa Hukum De Morgan yang '
           'kedua memang seperti itu. Mana yang benar dan kenapa?',
  'relevant_keywords': ['De Morgan', 'NOT', 'AND', 'OR', 'logika', 'tabel kebenaran'],
  'cognitive': '4TGI',
  'session_id': 'eval-027',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa salah menerapkan Hukum De Morgan kedua'},
 {'query': 'Di pseudocode CT, kalau saya tulis 7 / 2, hasilnya 3 atau 3.5? Di Python hasilnya 3.5 '
           'tapi di C hasilnya 3 karena integer division. Pseudocode CT pakai yang mana? Dan '
           'gimana cara nulis integer division di pseudocode?',
  'relevant_keywords': ['pembagian', 'integer', 'float', 'pseudocode', 'tipe data', 'modulo'],
  'cognitive': '3TAI',
  'session_id': 'eval-028',
  'query_type': 'gap',
  'context_note': 'Mahasiswa bingung konvensi pembagian integer vs float di pseudocode CT'},
 {'query': 'Tugas: buat tabel kebenaran untuk ekspresi (A AND NOT B) OR (NOT A AND B). Saya buat '
           'untuk semua kombinasi A dan B tapi hasilnya berbeda dengan teman. Saya dapat F,T,T,F '
           'tapi teman dapat T,T,T,F. Mana yang benar?',
  'relevant_keywords': ['tabel kebenaran', 'AND', 'OR', 'NOT', 'logika', 'evaluasi ekspresi'],
  'cognitive': '3PGI',
  'session_id': 'eval-029',
  'query_type': 'application',
  'context_note': 'Mahasiswa latihan tabel kebenaran XOR — ada di GT_SUBTOPIK_06'},
 {'query': 'Di materi variabel dan tipe data, kita belajar integer, float, boolean, string, dan '
           'list. Tapi bagaimana kalau saya mau simpan data mahasiswa yang punya nama, NIM, dan '
           'nilai sekaligus? Apakah ada tipe data gabungan seperti struct di CT?',
  'relevant_keywords': ['tipe data', 'variabel', 'list', 'string', 'struct', 'record'],
  'cognitive': '4PAI',
  'session_id': 'eval-030',
  'query_type': 'out_of_scope',
  'context_note': 'Tipe data struct/record tidak ada di GT CT — hanya ada primitif dan list'},
 {'query': 'Trace pseudocode ini untuk input x=-3, y=5:\n'
           '  if (x > 0) then\n'
           "    if (y > 0) then print('Q1') else print('Q4')\n"
           '  else\n'
           "    if (y > 0) then print('Q2') else print('Q3')\n"
           'Saya dapat Q2 tapi teman dapat Q3. Siapa yang benar?',
  'relevant_keywords': ['trace', 'percabangan', 'nested if', 'kondisi', 'output', 'kuadran'],
  'cognitive': '3PGR',
  'session_id': 'eval-031',
  'query_type': 'application',
  'context_note': 'Trace nested if percabangan kuadran ada di GT_SUBTOPIK_07'},
 {'query': 'Saya harus cek: apakah suhu > 35 DAN kelembaban > 80. Saya buat nested if — if suhu>35 '
           'then if kelembaban>80. Tapi teman pakai AND: if suhu>35 AND kelembaban>80. Kata dosen '
           'keduanya boleh tapi ada perbedaannya. Apa perbedaannya?',
  'relevant_keywords': ['nested if', 'AND', 'percabangan', 'kondisi', 'else-if', 'logika'],
  'cognitive': '4PAR',
  'session_id': 'eval-032',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa tidak tahu kapan nested if lebih tepat dari AND gabungan'},
 {'query': 'Di else-if untuk konversi nilai: A(>=85), B(>=70), C(>=55), D(lainnya). Nilai saya 90. '
           'Kondisi pertama (>=85) true, tapi kondisi kedua (>=70) juga true. Mana yang '
           'dieksekusi? Dan kenapa tidak keduanya sekaligus?',
  'relevant_keywords': ['else-if', 'kondisi', 'true', 'percabangan', 'sekuensial', 'eksekusi'],
  'cognitive': '2TAR',
  'session_id': 'eval-033',
  'query_type': 'gap',
  'context_note': 'Mahasiswa bingung kenapa hanya satu cabang yang dieksekusi di else-if'},
 {'query': 'Trace pseudocode konversi nilai untuk input nilai=68:\n'
           "  if (nilai>=85) then print('A')\n"
           "  else if (nilai>=70) then print('B')\n"
           "  else if (nilai>=55) then print('C')\n"
           "  else print('D')\n"
           'Saya dapat C tapi teman dapat D. Mana yang benar dan kenapa?',
  'relevant_keywords': ['trace', 'else-if', 'nilai', 'percabangan', 'kondisi', 'output'],
  'cognitive': '2PGI',
  'session_id': 'eval-034',
  'query_type': 'application',
  'context_note': 'Trace else-if konversi nilai — ada di GT_SUBTOPIK_07'},
 {'query': 'Saya mau buat program yang minta input terus sampai user masukkan angka valid (1-100). '
           'Ini perlu loop (while) dan percabangan (if untuk cek valid) sekaligus. Mana yang jadi '
           "'pembungkus' — apakah if di dalam while, atau while di dalam if?",
  'relevant_keywords': ['while', 'percabangan', 'validasi', 'loop', 'kondisi', 'sentinel'],
  'cognitive': '3TAR',
  'session_id': 'eval-035',
  'query_type': 'cross_topic',
  'context_note': 'Mahasiswa menggabungkan while dan if untuk validasi input'},
 {'query': 'Saya diminta baca semua baris file CSV sampai habis. Saya tidak tahu ada berapa baris '
           'di file itu. Apakah saya pakai for atau while? Dosen bilang ada kasus pakai for tapi '
           'ada yang pakai while untuk hal yang sama.',
  'relevant_keywords': ['for', 'while', 'iterasi', 'kondisi', 'file', 'sentinel'],
  'cognitive': '2PAI',
  'session_id': 'eval-036',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa bingung for vs while untuk membaca file dengan jumlah baris tidak '
                  'diketahui'},
 {'query': 'Saya diminta tulis pseudocode untuk cetak 10 bilangan genap pertama (2, 4, ..., 20) '
           'dalam DUA versi berbeda: satu dengan step dan satu tanpa step (pakai akumulasi). Versi '
           'dengan step saya sudah bisa, tapi yang akumulasi saya tidak mengerti maksudnya.',
  'relevant_keywords': ['for', 'step', 'genap', 'akumulasi', 'pseudocode', 'perulangan', 'while'],
  'cognitive': '2TGR',
  'session_id': 'eval-037',
  'query_type': 'application',
  'context_note': 'Pseudocode dua versi cetak bilangan genap ada di GT_SUBTOPIK_08'},
 {'query': "Di slide perulangan ada istilah 'sentinel value'. Katanya ini alternatif dari while "
           'biasa untuk baca input. Saya tidak paham apa itu sentinel dan kapan saya pakai itu '
           'daripada while biasa dengan kondisi boolean?',
  'relevant_keywords': ['sentinel', 'while', 'input', 'perulangan', 'kondisi', 'for'],
  'cognitive': '3PAI',
  'session_id': 'eval-038',
  'query_type': 'gap',
  'context_note': 'Mahasiswa tidak paham konsep sentinel value dalam perulangan'},
 {'query': 'Trace pseudocode berikut untuk input n=4523:\n'
           '  jumlah = 0\n'
           '  while (n > 0) do\n'
           '    jumlah = jumlah + (n mod 10)\n'
           '    n = n div 10\n'
           '  print(jumlah)\n'
           'Saya dapat 14 tapi teman dapat 41. Mana yang benar?',
  'relevant_keywords': ['trace', 'while', 'modulo', 'div', 'perulangan', 'digit'],
  'cognitive': '3TGR',
  'session_id': 'eval-039',
  'query_type': 'application',
  'context_note': 'Trace while loop hitung jumlah digit — ada di GT_SUBTOPIK_08'},
 {'query': 'Di C++ dan Java ada do-while loop yang eksekusi body setidaknya satu kali. Di '
           'pseudocode CT kita hanya belajar for dan while. Apakah do-while ada di pseudocode CT? '
           'Dan bagaimana simulasi do-while dengan while biasa?',
  'relevant_keywords': ['do-while', 'while', 'perulangan', 'body', 'kondisi', 'for'],
  'cognitive': '4TGI',
  'session_id': 'eval-040',
  'query_type': 'out_of_scope',
  'context_note': 'Do-while tidak ada di GT CT — hanya for dan while yang diajarkan'},
 {'query': 'Saya bisa buat segitiga bintang biasa (1 bintang di baris pertama, semakin banyak ke '
           'bawah). Tapi dosen minta buat yang TERBALIK: 5 bintang di baris pertama, semakin '
           'sedikit ke bawah. Saya tidak tahu harus ubah kondisi loop dalam-nya bagaimana.',
  'relevant_keywords': ['nested loop', 'segitiga', 'for', 'pola', 'bintang', 'baris'],
  'cognitive': '2PAR',
  'session_id': 'eval-041',
  'query_type': 'application',
  'context_note': 'Pola segitiga terbalik — variasi dari soal di GT_SUBTOPIK_09'},
 {'query': 'Saya punya nested loop: luar 5 iterasi, dalam 3 iterasi. Saya kira total iterasinya '
           '5+3=8. Tapi kata teman 5×3=15. Mana yang benar dan kenapa? Saya bingung karena selalu '
           'tambah di logika saya.',
  'relevant_keywords': ['nested loop', 'iterasi', 'total', 'perkalian', 'kompleksitas', 'O(n^2)'],
  'cognitive': '3PAR',
  'session_id': 'eval-042',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa salah mengira total iterasi nested loop adalah penjumlahan bukan '
                  'perkalian'},
 {'query': 'Tugas: cetak matriks identitas 3x3 dengan pseudocode nested loop. Saya tahu diagonal '
           'utamanya berisi 1 dan sisanya 0, tapi saya tidak tahu cara menulis kondisi untuk '
           'diagonal di pseudocode.',
  'relevant_keywords': ['nested loop', 'matriks', 'kondisi', 'diagonal', 'for', 'print'],
  'cognitive': '4PAI',
  'session_id': 'eval-043',
  'query_type': 'application',
  'context_note': 'Matriks identitas dengan nested loop — ada di GT_SUBTOPIK_09'},
 {'query': 'Dosen bilang nested loop selalu O(n²). Tapi saya punya nested loop: luar dari 1 to n, '
           'dalam dari 1 to 5 (konstan). Apakah itu masih O(n²)? Karena loop dalamnya tidak '
           'bergantung n.',
  'relevant_keywords': ['nested loop', 'O(n^2)', 'kompleksitas', 'konstan', 'Big-O', 'iterasi'],
  'cognitive': '5TGR',
  'session_id': 'eval-044',
  'query_type': 'gap',
  'context_note': 'Mahasiswa salah kira semua nested loop adalah O(n²)'},
 {'query': "Di Python, def bisa return nilai atau tidak — keduanya sama-sama 'fungsi'. Tapi di "
           "pseudocode CT, dosen bedakan 'fungsi' dan 'prosedur'. Apakah perbedaan ini penting di "
           'CT, atau hanya formalitas?',
  'relevant_keywords': ['fungsi', 'prosedur', 'return', 'Python', 'pseudocode', 'DRY'],
  'cognitive': '3TAR',
  'session_id': 'eval-045',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa Python tidak paham pembedaan fungsi vs prosedur di pseudocode CT'},
 {'query': 'Tugas: tulis tiga fungsi untuk luas persegi panjang, segitiga, dan lingkaran. Lalu '
           'buat program utama yang memanggil ketiganya. Saya bingung cara menulis parameter '
           'dengan tipe data di pseudocode CT.',
  'relevant_keywords': ['fungsi', 'parameter', 'return', 'luas', 'tipe data', 'prosedur'],
  'cognitive': '2PAI',
  'session_id': 'eval-046',
  'query_type': 'application',
  'context_note': 'Fungsi luas bangun datar ada di GT_SUBTOPIK_10 dan GT_DETAIL_PT12'},
 {'query': 'Saya punya variabel x = 10 di program utama. Di dalam fungsi saya ubah x = 99. Setelah '
           'fungsi selesai, nilai x di program utama jadi 99 atau tetap 10? Saya bingung karena di '
           'Python bisa berbeda hasilnya.',
  'relevant_keywords': ['variabel', 'scope', 'fungsi', 'lokal', 'global', 'return'],
  'cognitive': '3TAI',
  'session_id': 'eval-047',
  'query_type': 'gap',
  'context_note': 'Mahasiswa bingung scope variabel lokal vs global dalam fungsi pseudocode'},
 {'query': 'Rekursi itu fungsi yang memanggil dirinya sendiri. Tapi kalau rekursi adalah fungsi, '
           "kenapa diajarkan terpisah? Apa yang membuat rekursi 'spesial' dibanding fungsi biasa?",
  'relevant_keywords': ['rekursi', 'fungsi', 'self-call', 'base case', 'stack', 'prosedur'],
  'cognitive': '4TGI',
  'session_id': 'eval-048',
  'query_type': 'cross_topic',
  'context_note': 'Mahasiswa tidak paham apa yang membuat rekursi berbeda dari fungsi biasa'},
 {'query': 'Di Python ada konsep fungsi yang menerima fungsi lain sebagai parameter — disebut '
           'higher-order function. Apakah konsep ini diajarkan di CT? Atau CT hanya bahas fungsi '
           'yang menerima tipe data primitif?',
  'relevant_keywords': ['fungsi',
                        'parameter',
                        'higher-order',
                        'prosedur',
                        'abstraksi',
                        'modularitas'],
  'cognitive': '5PAI',
  'session_id': 'eval-049',
  'query_type': 'out_of_scope',
  'context_note': 'Higher-order function tidak ada di GT CT'},
 {'query': "Saya tulis fungsi rekursif tapi program saya crash dengan error 'maximum recursion "
           "depth exceeded'. Saya sudah tambahkan base case tapi masih crash. Apa yang salah? "
           'Apakah base case saya yang bermasalah atau ada hal lain?',
  'relevant_keywords': ['rekursi', 'base case', 'stack overflow', 'depth', 'infinite', 'fungsi'],
  'cognitive': '3PAI',
  'session_id': 'eval-050',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa punya rekursi yang crash — base case tidak memenuhi syarat'},
 {'query': 'Saya diminta trace faktorial(4) secara lengkap dan tunjukkan call stack di fase '
           "'turun' (push) dan fase 'naik' (pop). Saya bingung apa bedanya fase turun dan fase "
           'naik.',
  'relevant_keywords': ['faktorial', 'rekursi', 'call stack', 'trace', 'base case', 'push'],
  'cognitive': '3TGR',
  'session_id': 'eval-051',
  'query_type': 'application',
  'context_note': 'Trace faktorial rekursif dengan call stack ada di GT_SUBTOPIK_11'},
 {'query': 'Saya buat Fibonacci rekursif dan berjalan normal untuk Fib(10). Tapi waktu saya coba '
           'Fib(40) program sangat lambat — hampir 1 menit. Padahal logikanya sederhana. Kenapa '
           'bisa selambat itu?',
  'relevant_keywords': ['Fibonacci',
                        'rekursi',
                        'O(2^n)',
                        'kompleksitas',
                        'call stack',
                        'memoization'],
  'cognitive': '5TAR',
  'session_id': 'eval-052',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa tidak tahu mengapa Fibonacci rekursif eksponensial lambatnya'},
 {'query': 'Hitung Fib(6) menggunakan definisi rekursif: Fib(1)=1, Fib(2)=1, '
           'Fib(n)=Fib(n-1)+Fib(n-2). Saya dapat 11 tapi teman dapat 8. Tolong trace dari awal '
           'untuk membuktikan mana yang benar.',
  'relevant_keywords': ['Fibonacci', 'rekursi', 'trace', 'base case', 'perhitungan', 'deret'],
  'cognitive': '3TGI',
  'session_id': 'eval-053',
  'query_type': 'application',
  'context_note': 'Trace Fibonacci rekursif — ada di GT_SUBTOPIK_11'},
 {'query': 'Dosen bilang faktorial rekursif O(n) tapi Fibonacci rekursif O(2^n). Keduanya rekursi, '
           'kenapa kompleksitasnya bisa beda jauh? Apa yang menentukan kompleksitas dari sebuah '
           'fungsi rekursif?',
  'relevant_keywords': ['rekursi', 'kompleksitas', 'O(n)', 'O(2^n)', 'Fibonacci', 'faktorial'],
  'cognitive': '5TGI',
  'session_id': 'eval-054',
  'query_type': 'cross_topic',
  'context_note': 'Mahasiswa menghubungkan rekursi dengan analisis kompleksitas'},
 {'query': 'Di Python saya pakai list untuk segalanya dan tidak pernah ada masalah. Tapi di slide '
           'CT dibedakan antara array dan list. Apakah Python list itu array? Kalau iya, kenapa '
           'disebut beda?',
  'relevant_keywords': ['array', 'list', 'tipe data', 'memori', 'Python', 'indeks'],
  'cognitive': '4TGR',
  'session_id': 'eval-055',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa tidak paham perbedaan array klasik dan Python list'},
 {'query': 'Kalau Python list sudah bisa melakukan semua yang array bisa — tambah elemen, akses '
           'indeks, loop — kenapa orang masih pakai array? Kapan tepatnya array lebih baik dari '
           'list?',
  'relevant_keywords': ['array', 'list', 'efisiensi', 'memori', 'operasi', 'tipe data'],
  'cognitive': '5PAR',
  'session_id': 'eval-056',
  'query_type': 'gap',
  'context_note': 'Mahasiswa tidak paham keunggulan array dibanding dynamic list'},
 {'query': 'Saya baca tentang linked list di internet dan katanya lebih efisien dari array untuk '
           'insert dan delete. Apakah linked list diajarkan di mata kuliah CT ini? Kalau tidak, '
           'apa yang membedakan dengan array?',
  'relevant_keywords': ['linked list', 'array', 'insert', 'delete', 'struktur data', 'pointer'],
  'cognitive': '4TGI',
  'session_id': 'eval-057',
  'query_type': 'out_of_scope',
  'context_note': 'Linked list tidak ada di GT CT — hanya array/list yang diajarkan'},
 {'query': 'Saya punya list nilai [75, 82, 60, 91, 55, 88, 70]. Saya diminta hitung rata-rata, '
           'lalu tampilkan berapa mahasiswa di atas dan di bawah rata-rata. Apakah ini bisa '
           'diselesaikan dengan satu loop atau perlu dua loop?',
  'relevant_keywords': ['list', 'rata-rata', 'loop', 'for', 'akumulasi', 'perbandingan'],
  'cognitive': '3PAI',
  'session_id': 'eval-058',
  'query_type': 'application',
  'context_note': 'Operasi statistik pada list — ada di GT_DETAIL_PT09 dan GT_CT13'},
 {'query': 'Simulasikan stack dengan operasi berurutan:\n'
           'PUSH(7), PUSH(2), PUSH(5), POP, PUSH(9), PEEK, POP, POP.\n'
           'Saya tidak yakin apa yang dikembalikan PEEK — apakah elemen terhapus atau tidak?',
  'relevant_keywords': ['stack', 'PUSH', 'POP', 'PEEK', 'LIFO', 'top', 'simulasi'],
  'cognitive': '2TGI',
  'session_id': 'eval-059',
  'query_type': 'application',
  'context_note': 'Simulasi operasi stack — ada di GT_SUBTOPIK_12'},
 {'query': 'Simulasikan queue dengan operasi:\n'
           'ENQUEUE(P), ENQUEUE(Q), ENQUEUE(R), DEQUEUE, ENQUEUE(S), DEQUEUE, DEQUEUE.\n'
           'Tunjukkan isi queue setelah setiap operasi dan nilai yang dikembalikan DEQUEUE.',
  'relevant_keywords': ['queue', 'ENQUEUE', 'DEQUEUE', 'FIFO', 'front', 'rear', 'simulasi'],
  'cognitive': '2PAR',
  'session_id': 'eval-060',
  'query_type': 'application',
  'context_note': 'Simulasi operasi queue — ada di GT_SUBTOPIK_12'},
 {'query': 'Saya mau buat fitur undo di aplikasi teks editor saya. Setiap aksi disimpan dan bisa '
           'di-undo ke aksi sebelumnya. Apakah saya pakai stack atau queue? Dan mengapa?',
  'relevant_keywords': ['stack', 'queue', 'LIFO', 'FIFO', 'undo', 'aplikasi'],
  'cognitive': '3TAI',
  'session_id': 'eval-061',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa memilih stack vs queue untuk fitur undo'},
 {'query': "Soal: periksa apakah ekspresi '((a+b)*(c-d))' seimbang kurungnya. Dosen minta pakai "
           'stack. Saya paham LIFO tapi tidak tahu cara terapkan ke masalah kurung ini.',
  'relevant_keywords': ['stack', 'kurung', 'PUSH', 'POP', 'seimbang', 'LIFO', 'ekspresi'],
  'cognitive': '5TGI',
  'session_id': 'eval-062',
  'query_type': 'application',
  'context_note': 'Aplikasi stack untuk cek keseimbangan kurung — ada di GT_SUBTOPIK_12'},
 {'query': "Di slide stack ada pseudocode implementasi dengan array dan variabel 'top'. Saya tidak "
           'mengerti mengapa perlu variabel top — kenapa tidak langsung pakai panjang array? Apa '
           'fungsi variabel top dalam implementasi stack?',
  'relevant_keywords': ['stack', 'implementasi', 'array', 'top', 'PUSH', 'POP'],
  'cognitive': '4TGR',
  'session_id': 'eval-063',
  'query_type': 'gap',
  'context_note': 'Mahasiswa tidak paham peran variabel top dalam implementasi stack berbasis '
                  'array'},
 {'query': "Dosen bilang tree itu 'graph khusus'. Tapi kalau tree adalah graph, kenapa diajarkan "
           'terpisah? Apa yang membuat graph dianggap tree dan bukan hanya graph biasa?',
  'relevant_keywords': ['tree', 'graph', 'cycle', 'hierarkis', 'node', 'edge'],
  'cognitive': '4TGI',
  'session_id': 'eval-064',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa bingung hubungan hierarki antara tree dan graph'},
 {'query': 'Graph: A-B(4), A-C(2), B-D(3), C-D(5), D-E(1). Cari jarak terpendek dari A ke E '
           'menggunakan Dijkstra. Saya sudah coba tapi selalu dapat jalur yang berbeda dari teman.',
  'relevant_keywords': ['Dijkstra', 'graph', 'jarak terpendek', 'bobot', 'BFS', 'traversal'],
  'cognitive': '3TGI',
  'session_id': 'eval-065',
  'query_type': 'application',
  'context_note': 'Trace Dijkstra ada di GT_SUBTOPIK_13'},
 {'query': 'Graph: A→B, A→C, B→D, B→E, C→F. Saya trace BFS dari A dan dapat: A, B, C, D, E, F. '
           'Teman trace DFS dari A dan dapat: A, B, D, E, C, F. Apakah keduanya benar? Dan kenapa '
           'urutannya bisa berbeda?',
  'relevant_keywords': ['BFS', 'DFS', 'traversal', 'queue', 'stack', 'urutan', 'graph'],
  'cognitive': '4TGI',
  'session_id': 'eval-066',
  'query_type': 'comparative',
  'context_note': 'Perbandingan BFS vs DFS ada di GT_SUBTOPIK_13'},
 {'query': 'Di slide graph ada directed dan undirected. Dosen minta pilih jenis graph yang tepat '
           "untuk 'jaringan pertemanan di media sosial'. Saya tidak yakin — pertemanan kan "
           'simetris tapi ada yang following tanpa difollow balik. Mana yang benar?',
  'relevant_keywords': ['directed graph',
                        'undirected graph',
                        'edge',
                        'simetris',
                        'media sosial',
                        'relasi'],
  'cognitive': '3PAR',
  'session_id': 'eval-067',
  'query_type': 'gap',
  'context_note': 'Memilih directed vs undirected graph untuk kasus nyata — ada di GT_SUBTOPIK_13'},
 {'query': 'Saya baca tentang Binary Search Tree (BST) di internet. Apakah BST diajarkan di CT? '
           'Kalau ya, bagaimana cara insert dan search di BST?',
  'relevant_keywords': ['binary search tree', 'tree', 'insert', 'search', 'node', 'hierarkis'],
  'cognitive': '5PGR',
  'session_id': 'eval-068',
  'query_type': 'out_of_scope',
  'context_note': 'BST operasi spesifik tidak ada di GT CT — hanya konsep tree dasar'},
 {'query': 'Tentukan kompleksitas waktu dari:\n'
           'a) x = arr[5]\n'
           'b) for i=1 to n: print(arr[i])\n'
           'c) for i=1 to n: for j=1 to n: print(i+j)\n'
           'Saya dapat a=O(1), b=O(n), c=O(n²). Apakah benar?',
  'relevant_keywords': ['Big-O', 'O(1)', 'O(n)', 'O(n^2)', 'loop', 'nested loop', 'kompleksitas'],
  'cognitive': '3PAI',
  'session_id': 'eval-069',
  'query_type': 'application',
  'context_note': 'Identifikasi Big-O dari kode ada di GT_SUBTOPIK_14'},
 {'query': 'Program saya punya satu nested loop O(n²) dan satu loop biasa O(n) yang berjalan '
           'setelahnya. Saya tulis kompleksitasnya O(n²+n). Tapi dosen coret itu dan tulis O(n²). '
           'Kenapa n-nya dibuang?',
  'relevant_keywords': ['Big-O', 'penyederhanaan', 'O(n^2)', 'O(n)', 'suku dominan', 'aturan'],
  'cognitive': '4PAR',
  'session_id': 'eval-070',
  'query_type': 'confusion',
  'context_note': 'Mahasiswa bingung aturan penyederhanaan Big-O — buang suku rendah'},
 {'query': 'Tentukan Big-O pseudocode ini:\n'
           '  for i=1 to n do\n'
           '    j=1\n'
           '    while (j < n) do\n'
           '      print(i,j)\n'
           '      j = j*2\n'
           'Saya kira jawabannya O(n²) karena ada nested loop. Tapi dosen bilang bukan.',
  'relevant_keywords': ['Big-O', 'O(n log n)', 'while', 'logaritmik', 'pengali', 'analisis'],
  'cognitive': '5TGI',
  'session_id': 'eval-071',
  'query_type': 'application',
  'context_note': 'Big-O loop dengan j*=2 ada di GT_SUBTOPIK_14'},
 {'query': 'Saya punya 100.000 data yang perlu diurutkan. Bubble sort terasa lambat. Dosen '
           'menyarankan merge sort. Kalau keduanya bisa menghasilkan array terurut, kenapa merge '
           'sort lebih baik untuk data besar?',
  'relevant_keywords': ['bubble sort',
                        'merge sort',
                        'O(n^2)',
                        'O(n log n)',
                        'kompleksitas',
                        'sorting'],
  'cognitive': '5PAR',
  'session_id': 'eval-072',
  'query_type': 'comparative',
  'context_note': 'Perbandingan sorting berdasarkan kompleksitas ada di GT_SUBTOPIK_14'},
 {'query': 'Saya hanya tahu time complexity. Tapi dosen juga tanya space complexity dari algoritme '
           'saya. Apa itu space complexity dan bagaimana cara menganalisisnya? Apakah O(1) space '
           'berarti tidak pakai memori sama sekali?',
  'relevant_keywords': ['space complexity', 'memori', 'O(1)', 'O(n)', 'algoritme', 'rekursi'],
  'cognitive': '4TGR',
  'session_id': 'eval-073',
  'query_type': 'gap',
  'context_note': 'Space complexity dibahas di GT_SUBTOPIK_14 dalam konteks time vs space'},
 {'query': 'Saya baca bahwa append ke Python list itu O(1) amortized, bukan O(1) biasa. Apa itu '
           'amortized complexity dan kenapa append tidak selalu O(1)?',
  'relevant_keywords': ['amortized', 'append', 'list', 'O(1)', 'kompleksitas', 'array'],
  'cognitive': '6TGR',
  'session_id': 'eval-074',
  'query_type': 'out_of_scope',
  'context_note': 'Amortized complexity tidak ada di GT CT'},
 {'query': 'Array terurut: [3, 7, 12, 18, 25, 31, 44, 56, 67, 89]. Cari angka 25. Saya trace '
           'binary search dan butuh 4 langkah. Teman hanya butuh 2 langkah. Tolong trace dari awal '
           'untuk verifikasi berapa langkah yang benar.',
  'relevant_keywords': ['binary search', 'trace', 'low', 'high', 'mid', 'langkah', 'array terurut'],
  'cognitive': '3TGR',
  'session_id': 'eval-075',
  'query_type': 'application',
  'context_note': 'Trace binary search step by step'}]
