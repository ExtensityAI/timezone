from symai import Expression, Function
import numpy as np
from difflib import SequenceMatcher


FUNCTION_DESCRIPTION = '''Convert between time zones based on the country and time table below.
Elaborate on the countries and time zones that are not included in the table.
Do A step-by-step computation of the difference and then reply based on the user request.

List
-----------------
Country code(s)	TZ identifier	Embedded comments	Type	UTC offset	Time zone	Source
				±hh:mm	abbreviation	file   STD	DST	STD	DST
-------------------------------------------------------------------------------------
	CET		Canonical	+01:00	+02:00	CET
	CST6CDT		Canonical	−06:00	−05:00	CST
	EET		Canonical	+02:00	+03:00	EET
	EST		Canonical	−05:00	−05:00	EST
	EST5EDT		Canonical	−05:00	−04:00	EST
	Etc/GMT		Canonical	+00:00	+00:00	GMT
	Etc/GMT+0		Link	+00:00	+00:00	GMT
	Etc/GMT+1		Canonical	−01:00	−01:00	-1
	Etc/GMT+10		Canonical	−10:00	−10:00	-10
	Etc/GMT+11		Canonical	−11:00	−11:00	-11
	Etc/GMT+12		Canonical	−12:00	−12:00	-12
	Etc/GMT+2		Canonical	−02:00	−02:00	-2
	Etc/GMT+3		Canonical	−03:00	−03:00	-3
	Etc/GMT+4		Canonical	−04:00	−04:00	-4
	Etc/GMT+5		Canonical	−05:00	−05:00	-5
	Etc/GMT+6		Canonical	−06:00	−06:00	-6
	Etc/GMT+7		Canonical	−07:00	−07:00	-7
	Etc/GMT+8		Canonical	−08:00	−08:00	-8
	Etc/GMT+9		Canonical	−09:00	−09:00	-9
	Etc/GMT-0		Link	+00:00	+00:00	GMT
	Etc/GMT-1		Canonical	+01:00	+01:00	1
	Etc/GMT-10		Canonical	+10:00	+10:00	10
	Etc/GMT-11		Canonical	+11:00	+11:00	11
	Etc/GMT-12		Canonical	+12:00	+12:00	12
	Etc/GMT-13		Canonical	+13:00	+13:00	13
	Etc/GMT-14		Canonical	+14:00	+14:00	14
	Etc/GMT-2		Canonical	+02:00	+02:00	2
	Etc/GMT-3		Canonical	+03:00	+03:00	3
	Etc/GMT-4		Canonical	+04:00	+04:00	4
	Etc/GMT-5		Canonical	+05:00	+05:00	5
	Etc/GMT-6		Canonical	+06:00	+06:00	6
	Etc/GMT-7		Canonical	+07:00	+07:00	7
	Etc/GMT-8		Canonical	+08:00	+08:00	8
	Etc/GMT-9		Canonical	+09:00	+09:00	9
	Etc/GMT0		Link	+00:00	+00:00	GMT
	Etc/Greenwich		Link	+00:00	+00:00	GMT
	Etc/UCT		Link	+00:00	+00:00	UTC
	Etc/Universal		Link	+00:00	+00:00	UTC
	Etc/UTC		Canonical	+00:00	+00:00	UTC
	Etc/Zulu		Link	+00:00	+00:00	UTC
	Factory		Canonical	+00:00	+00:00	0
	GMT		Link	+00:00	+00:00	GMT
	GMT+0		Link	+00:00	+00:00	GMT
	GMT-0		Link	+00:00	+00:00	GMT
	GMT0		Link	+00:00	+00:00	GMT
	Greenwich		Link	+00:00	+00:00	GMT
	HST		Canonical	−10:00	−10:00	HST
	MET		Canonical	+01:00	+02:00	MET
	MST		Canonical	−07:00	−07:00	MST
	MST7MDT		Canonical	−07:00	−06:00	MST
	PST8PDT		Canonical	−08:00	−07:00	PST
	UCT		Link	+00:00	+00:00	UTC
	Universal		Link	+00:00	+00:00	UTC
	UTC		Link	+00:00	+00:00	UTC
	WET		Canonical	+00:00	+01:00	WET
	Zulu		Link	+00:00	+00:00	UTC
'''


COUNTRY_TIME_TABLE = '''
AD	Europe/Andorra		Canonical	+01:00	+02:00	CET
AE, OM, RE, SC, TF	Asia/Dubai	Crozet, Scattered Is	Canonical	+04:00	+04:00	4
AF	Asia/Kabul		Canonical	+04:30	+04:30	430
AG	America/Antigua		Link†	−04:00	−04:00	AST
AI	America/Anguilla		Link†	−04:00	−04:00	AST
AL	Europe/Tirane		Canonical	+01:00	+02:00	CET
AM	Asia/Yerevan		Canonical	+04:00	+04:00	4
AO	Africa/Luanda		Link†	+01:00	+01:00	WAT
AQ	Antarctica/Casey	Casey	Canonical	+11:00	+11:00	11
AQ	Antarctica/Davis	Davis	Canonical	+07:00	+07:00	7
AQ	Antarctica/DumontDUrville	Dumont-d'Urville	Link†	+10:00	+10:00	10
AQ	Antarctica/Mawson	Mawson	Canonical	+05:00	+05:00	5
AQ	Antarctica/McMurdo	New Zealand time - McMurdo, South Pole	Link†	+12:00	+13:00	NZST
AQ	Antarctica/Palmer	Palmer	Canonical	−03:00	−03:00	-3
AQ	Antarctica/Rothera	Rothera	Canonical	−03:00	−03:00	-3
AQ	Antarctica/South_Pole		Link	+12:00	+13:00	NZST
AQ	Antarctica/Syowa	Syowa	Link†	+03:00	+03:00	3
AQ	Antarctica/Troll	Troll	Canonical	+00:00	+02:00	0
AQ	Antarctica/Vostok	Vostok	Link†	+06:00	+06:00	6
AR	America/Argentina/Buenos_Aires	Buenos Aires (BA, CF)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/Catamarca	Catamarca (CT); Chubut (CH)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/ComodRivadavia		Link†	−03:00	−03:00	-3
AR	America/Argentina/Cordoba	most areas: CB, CC, CN, ER, FM, MN, SE, SF	Canonical	−03:00	−03:00	-3
AR	America/Argentina/Jujuy	Jujuy (JY)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/La_Rioja	La Rioja (LR)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/Mendoza	Mendoza (MZ)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/Rio_Gallegos	Santa Cruz (SC)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/Salta	Salta (SA, LP, NQ, RN)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/San_Juan	San Juan (SJ)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/San_Luis	San Luis (SL)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/Tucuman	Tucumán (TM)	Canonical	−03:00	−03:00	-3
AR	America/Argentina/Ushuaia	Tierra del Fuego (TF)	Canonical	−03:00	−03:00	-3
AR	America/Buenos_Aires		Link	−03:00	−03:00	-3
AR	America/Catamarca		Link	−03:00	−03:00	-3
AR	America/Cordoba		Link	−03:00	−03:00	-3
AR	America/Jujuy		Link	−03:00	−03:00	-3
AR	America/Mendoza		Link	−03:00	−03:00	-3
AR	America/Rosario		Link†	−03:00	−03:00	-3
AS, UM	Pacific/Pago_Pago	Midway	Canonical	−11:00	−11:00	SST
AS	Pacific/Samoa		Link	−11:00	−11:00	SST
AS	US/Samoa		Link	−11:00	−11:00	SST
AT	Europe/Vienna		Canonical	+01:00	+02:00	CET
AU	Antarctica/Macquarie	Macquarie Island	Canonical	+10:00	+11:00	AEST
AU	Australia/ACT		Link	+10:00	+11:00	AEST
AU	Australia/Adelaide	South Australia	Canonical	+09:30	+10:30	ACST
AU	Australia/Brisbane	Queensland (most areas)	Canonical	+10:00	+10:00	AEST
AU	Australia/Broken_Hill	New South Wales (Yancowinna)	Canonical	+09:30	+10:30	ACST
AU	Australia/Canberra		Link	+10:00	+11:00	AEST
AU	Australia/Currie		Link†	+10:00	+11:00	AEST
AU	Australia/Darwin	Northern Territory	Canonical	+09:30	+09:30	ACST
AU	Australia/Eucla	Western Australia (Eucla)	Canonical	+08:45	+08:45	845
AU	Australia/Hobart	Tasmania	Canonical	+10:00	+11:00	AEST
AU	Australia/LHI		Link	+10:30	+11:00	1030
AU	Australia/Lindeman	Queensland (Whitsunday Islands)	Canonical	+10:00	+10:00	AEST
AU	Australia/Lord_Howe	Lord Howe Island	Canonical	+10:30	+11:00	1030
AU	Australia/Melbourne	Victoria	Canonical	+10:00	+11:00	AEST
AU	Australia/North		Link	+09:30	+09:30	ACST
AU	Australia/NSW		Link	+10:00	+11:00	AEST
AU	Australia/Perth	Western Australia (most areas)	Canonical	+08:00	+08:00	AWST
AU	Australia/Queensland		Link	+10:00	+10:00	AEST
AU	Australia/South		Link	+09:30	+10:30	ACST
AU	Australia/Sydney	New South Wales (most areas)	Canonical	+10:00	+11:00	AEST
AU	Australia/Tasmania		Link	+10:00	+11:00	AEST
AU	Australia/Victoria		Link	+10:00	+11:00	AEST
AU	Australia/West		Link	+08:00	+08:00	AWST
AU	Australia/Yancowinna		Link	+09:30	+10:30	ACST
AW	America/Aruba		Link†	−04:00	−04:00	AST
AX	Europe/Mariehamn		Link	+02:00	+03:00	EET
AZ	Asia/Baku		Canonical	+04:00	+04:00	4
BA	Europe/Sarajevo		Link†	+01:00	+02:00	CET
BB	America/Barbados		Canonical	−04:00	−04:00	AST
BD	Asia/Dacca		Link	+06:00	+06:00	6
BD	Asia/Dhaka		Canonical	+06:00	+06:00	6
BE, LU, NL	Europe/Brussels		Canonical	+01:00	+02:00	CET
BF	Africa/Ouagadougou		Link†	+00:00	+00:00	GMT
BG	Europe/Sofia		Canonical	+02:00	+03:00	EET
BH	Asia/Bahrain		Link†	+03:00	+03:00	3
BI	Africa/Bujumbura		Link†	+02:00	+02:00	CAT
BJ	Africa/Porto-Novo		Link†	+01:00	+01:00	WAT
BL	America/St_Barthelemy		Link	−04:00	−04:00	AST
BM	Atlantic/Bermuda		Canonical	−04:00	−03:00	AST
BN	Asia/Brunei		Link†	+08:00	+08:00	8
BO	America/La_Paz		Canonical	−04:00	−04:00	-4
BQ	America/Kralendijk		Link	−04:00	−04:00	AST
BR	America/Araguaina	Tocantins	Canonical	−03:00	−03:00	-3
BR	America/Bahia	Bahia	Canonical	−03:00	−03:00	-3
BR	America/Belem	Pará (east); Amapá	Canonical	−03:00	−03:00	-3
BR	America/Boa_Vista	Roraima	Canonical	−04:00	−04:00	-4
BR	America/Campo_Grande	Mato Grosso do Sul	Canonical	−04:00	−04:00	-4
BR	America/Cuiaba	Mato Grosso	Canonical	−04:00	−04:00	-4
BR	America/Eirunepe	Amazonas (west)	Canonical	−05:00	−05:00	-5
BR	America/Fortaleza	Brazil (northeast: MA, PI, CE, RN, PB)	Canonical	−03:00	−03:00	-3
BR	America/Maceio	Alagoas, Sergipe	Canonical	−03:00	−03:00	-3
BR	America/Manaus	Amazonas (east)	Canonical	−04:00	−04:00	-4
BR	America/Noronha	Atlantic islands	Canonical	−02:00	−02:00	-2
BR	America/Porto_Acre		Link	−05:00	−05:00	-5
BR	America/Porto_Velho	Rondônia	Canonical	−04:00	−04:00	-4
BR	America/Recife	Pernambuco	Canonical	−03:00	−03:00	-3
BR	America/Rio_Branco	Acre	Canonical	−05:00	−05:00	-5
BR	America/Santarem	Pará (west)	Canonical	−03:00	−03:00	-3
BR	America/Sao_Paulo	Brazil (southeast: GO, DF, MG, ES, RJ, SP, PR, SC, RS)	Canonical	−03:00	−03:00	-3
BR	Brazil/Acre		Link	−05:00	−05:00	-5
BR	Brazil/DeNoronha		Link	−02:00	−02:00	-2
BR	Brazil/East		Link	−03:00	−03:00	-3
BR	Brazil/West		Link	−04:00	−04:00	-4
BS	America/Nassau		Link†	−05:00	−04:00	EST
BT	Asia/Thimbu		Link	+06:00	+06:00	6
BT	Asia/Thimphu		Canonical	+06:00	+06:00	6
BW	Africa/Gaborone		Link†	+02:00	+02:00	CAT
BY	Europe/Minsk		Canonical	+03:00	+03:00	3
BZ	America/Belize		Canonical	−06:00	−06:00	CST
CA	America/Atikokan	EST - ON (Atikokan); NU (Coral H)	Link†	−05:00	−05:00	EST
CA	America/Blanc-Sablon	AST - QC (Lower North Shore)	Link†	−04:00	−04:00	AST
CA	America/Cambridge_Bay	Mountain - NU (west)	Canonical	−07:00	−06:00	MST
CA	America/Coral_Harbour		Link†	−05:00	−05:00	EST
CA	America/Creston	MST - BC (Creston)	Link†	−07:00	−07:00	MST
CA	America/Dawson	MST - Yukon (west)	Canonical	−07:00	−07:00	MST
CA	America/Dawson_Creek	MST - BC (Dawson Cr, Ft St John)	Canonical	−07:00	−07:00	MST
CA	America/Edmonton	Mountain - AB; BC (E); NT (E); SK (W)	Canonical	−07:00	−06:00	MST
CA	America/Fort_Nelson	MST - BC (Ft Nelson)	Canonical	−07:00	−07:00	MST
CA	America/Glace_Bay	Atlantic - NS (Cape Breton)	Canonical	−04:00	−03:00	AST
CA	America/Goose_Bay	Atlantic - Labrador (most areas)	Canonical	−04:00	−03:00	AST
CA	America/Halifax	Atlantic - NS (most areas); PE	Canonical	−04:00	−03:00	AST
CA	America/Inuvik	Mountain - NT (west)	Canonical	−07:00	−06:00	MST
CA	America/Iqaluit	Eastern - NU (most areas)	Canonical	−05:00	−04:00	EST
CA	America/Moncton	Atlantic - New Brunswick	Canonical	−04:00	−03:00	AST
CA	America/Montreal		Link†	−05:00	−04:00	EST
CA	America/Nipigon		Link†	−05:00	−04:00	EST
CA	America/Pangnirtung		Link†	−05:00	−04:00	EST
CA	America/Rainy_River		Link†	−06:00	−05:00	CST
CA	America/Rankin_Inlet	Central - NU (central)	Canonical	−06:00	−05:00	CST
CA	America/Regina	CST - SK (most areas)	Canonical	−06:00	−06:00	CST
CA	America/Resolute	Central - NU (Resolute)	Canonical	−06:00	−05:00	CST
CA	America/St_Johns	Newfoundland; Labrador (southeast)	Canonical	−03:30	−02:30	NST
CA	America/Swift_Current	CST - SK (midwest)	Canonical	−06:00	−06:00	CST
CA	America/Thunder_Bay		Link†	−05:00	−04:00	EST
CA, BS	America/Toronto	Eastern - ON, QC (most areas)	Canonical	−05:00	−04:00	EST
CA	America/Vancouver	Pacific - BC (most areas)	Canonical	−08:00	−07:00	PST
CA	America/Whitehorse	MST - Yukon (east)	Canonical	−07:00	−07:00	MST
CA	America/Winnipeg	Central - ON (west); Manitoba	Canonical	−06:00	−05:00	CST
CA	America/Yellowknife		Link†	−07:00	−06:00	MST
CA	Canada/Atlantic		Link	−04:00	−03:00	AST
CA	Canada/Central		Link	−06:00	−05:00	CST
CA	Canada/Eastern		Link	−05:00	−04:00	EST
CA	Canada/Mountain		Link	−07:00	−06:00	MST
CA	Canada/Newfoundland		Link	−03:30	−02:30	NST
CA	Canada/Pacific		Link	−08:00	−07:00	PST
CA	Canada/Saskatchewan		Link	−06:00	−06:00	CST
CA	Canada/Yukon		Link	−07:00	−07:00	MST
CC	Indian/Cocos		Link†	+06:30	+06:30	630
CD	Africa/Kinshasa	Dem. Rep. of Congo (west)	Link†	+01:00	+01:00	WAT
CD	Africa/Lubumbashi	Dem. Rep. of Congo (east)	Link†	+02:00	+02:00	CAT
CF	Africa/Bangui		Link†	+01:00	+01:00	WAT
CG	Africa/Brazzaville		Link†	+01:00	+01:00	WAT
CH, DE, LI	Europe/Zurich	Büsingen	Canonical	+01:00	+02:00	CET
CI, BF, GH, GM, GN, IS, ML, MR, SH, SL, SN, TG	Africa/Abidjan		Canonical	+00:00	+00:00	GMT
CK	Pacific/Rarotonga		Canonical	−10:00	−10:00	-10
CL	America/Punta_Arenas	Region of Magallanes	Canonical	−03:00	−03:00	-3
CL	America/Santiago	most of Chile	Canonical	−04:00	−03:00	-4
CL	Chile/Continental		Link	−04:00	−03:00	-4
CL	Chile/EasterIsland		Link	−06:00	−05:00	-6
CL	Pacific/Easter	Easter Island	Canonical	−06:00	−05:00	-6
CM	Africa/Douala		Link†	+01:00	+01:00	WAT
CN	Asia/Chongqing		Link†	+08:00	+08:00	CST
CN	Asia/Chungking		Link	+08:00	+08:00	CST
CN	Asia/Harbin		Link†	+08:00	+08:00	CST
CN	Asia/Kashgar		Link†	+06:00	+06:00	6
CN	Asia/Shanghai	Beijing Time	Canonical	+08:00	+08:00	CST
CN, AQ	Asia/Urumqi	Xinjiang Time, Vostok	Canonical	+06:00	+06:00	6
CN	PRC		Link	+08:00	+08:00	CST
CO	America/Bogota		Canonical	−05:00	−05:00	-5
CR	America/Costa_Rica		Canonical	−06:00	−06:00	CST
CU	America/Havana		Canonical	−05:00	−04:00	CST
CU	Cuba		Link	−05:00	−04:00	CST
CV	Atlantic/Cape_Verde		Canonical	−01:00	−01:00	-1
CW	America/Curacao		Link†	−04:00	−04:00	AST
CX	Indian/Christmas		Link†	+07:00	+07:00	7
CY	Asia/Famagusta	Northern Cyprus	Canonical	+02:00	+03:00	EET
CY	Asia/Nicosia	most of Cyprus	Canonical	+02:00	+03:00	EET
CY	Europe/Nicosia		Link	+02:00	+03:00	EET
CZ, SK	Europe/Prague		Canonical	+01:00	+02:00	CET
DE, DK, NO, SE, SJ	Europe/Berlin	most of Germany	Canonical	+01:00	+02:00	CET
DE	Europe/Busingen	Busingen	Link	+01:00	+02:00	CET
DJ	Africa/Djibouti		Link†	+03:00	+03:00	EAT
DK	Europe/Copenhagen		Link†	+01:00	+02:00	CET
DM	America/Dominica		Link†	−04:00	−04:00	AST
DO	America/Santo_Domingo		Canonical	−04:00	−04:00	AST
DZ	Africa/Algiers		Canonical	+01:00	+01:00	CET
EC	America/Guayaquil	Ecuador (mainland)	Canonical	−05:00	−05:00	-5
EC	Pacific/Galapagos	Galápagos Islands	Canonical	−06:00	−06:00	-6
EE	Europe/Tallinn		Canonical	+02:00	+03:00	EET
EG	Africa/Cairo		Canonical	+02:00	+03:00	EET
EG	Egypt		Link	+02:00	+03:00	EET
EH	Africa/El_Aaiun		Canonical	+01:00	+00:00	1
ER	Africa/Asmara		Link†	+03:00	+03:00	EAT
ER	Africa/Asmera		Link	+03:00	+03:00	EAT
ES	Africa/Ceuta	Ceuta, Melilla	Canonical	+01:00	+02:00	CET
ES	Atlantic/Canary	Canary Islands	Canonical	+00:00	+01:00	WET
ES	Europe/Madrid	Spain (mainland)	Canonical	+01:00	+02:00	CET
ET	Africa/Addis_Ababa		Link†	+03:00	+03:00	EAT
FI, AX	Europe/Helsinki		Canonical	+02:00	+03:00	EET
FJ	Pacific/Fiji		Canonical	+12:00	+12:00	12
FK	Atlantic/Stanley		Canonical	−03:00	−03:00	-3
FM	Pacific/Chuuk	Chuuk/Truk, Yap	Link†	+10:00	+10:00	10
FM	Pacific/Kosrae	Kosrae	Canonical	+11:00	+11:00	11
FM	Pacific/Pohnpei	Pohnpei/Ponape	Link†	+11:00	+11:00	11
FM	Pacific/Ponape		Link	+11:00	+11:00	11
FM	Pacific/Truk		Link	+10:00	+10:00	10
FM	Pacific/Yap		Link	+10:00	+10:00	10
FO	Atlantic/Faeroe		Link	+00:00	+01:00	WET
FO	Atlantic/Faroe		Canonical	+00:00	+01:00	WET
FR, MC	Europe/Paris		Canonical	+01:00	+02:00	CET
GA	Africa/Libreville		Link†	+01:00	+01:00	WAT
GB	Europe/Belfast		Link†	+00:00	+01:00	GMT
GB, GG, IM, JE	Europe/London		Canonical	+00:00	+01:00	GMT
GB	GB		Link	+00:00	+01:00	GMT
GB	GB-Eire		Link	+00:00	+01:00	GMT
GD	America/Grenada		Link†	−04:00	−04:00	AST
GE	Asia/Tbilisi		Canonical	+04:00	+04:00	4
GF	America/Cayenne		Canonical	−03:00	−03:00	-3
GG	Europe/Guernsey		Link†	+00:00	+01:00	GMT
GH	Africa/Accra		Link†	+00:00	+00:00	GMT
GI	Europe/Gibraltar		Canonical	+01:00	+02:00	CET
GL	America/Danmarkshavn	National Park (east coast)	Canonical	+00:00	+00:00	GMT
GL	America/Godthab		Link	−02:00	−02:00	-2
GL	America/Nuuk	most of Greenland	Canonical	−02:00	−02:00	-2
GL	America/Scoresbysund	Scoresbysund/Ittoqqortoormiit	Canonical	−01:00	+00:00	-1
GL	America/Thule	Thule/Pituffik	Canonical	−04:00	−03:00	AST
GM	Africa/Banjul		Link†	+00:00	+00:00	GMT
GN	Africa/Conakry		Link†	+00:00	+00:00	GMT
GP	America/Guadeloupe		Link†	−04:00	−04:00	AST
GQ	Africa/Malabo		Link†	+01:00	+01:00	WAT
GR	Europe/Athens		Canonical	+02:00	+03:00	EET
GS	Atlantic/South_Georgia		Canonical	−02:00	−02:00	-2
GT	America/Guatemala		Canonical	−06:00	−06:00	CST
GU, MP	Pacific/Guam		Canonical	+10:00	+10:00	ChST
GW	Africa/Bissau		Canonical	+00:00	+00:00	GMT
GY	America/Guyana		Canonical	−04:00	−04:00	-4
HK	Asia/Hong_Kong		Canonical	+08:00	+08:00	HKT
HK	Hongkong		Link	+08:00	+08:00	HKT
HN	America/Tegucigalpa		Canonical	−06:00	−06:00	CST
HR	Europe/Zagreb		Link†	+01:00	+02:00	CET
HT	America/Port-au-Prince		Canonical	−05:00	−04:00	EST
HU	Europe/Budapest		Canonical	+01:00	+02:00	CET
ID	Asia/Jakarta	Java, Sumatra	Canonical	+07:00	+07:00	WIB
ID	Asia/Jayapura	New Guinea (West Papua / Irian Jaya); Malukus/Moluccas	Canonical	+09:00	+09:00	WIT
ID	Asia/Makassar	Borneo (east, south); Sulawesi/Celebes, Bali, Nusa Tengarra; Timor (west)	Canonical	+08:00	+08:00	WITA
ID	Asia/Pontianak	Borneo (west, central)	Canonical	+07:00	+07:00	WIB
ID	Asia/Ujung_Pandang		Link	+08:00	+08:00	WITA
IE	Eire		Link	+01:00	+00:00	IST
IE	Europe/Dublin		Canonical	+01:00	+00:00	IST
IL	Asia/Jerusalem		Canonical	+02:00	+03:00	IST
IL	Asia/Tel_Aviv		Link†	+02:00	+03:00	IST
IL	Israel		Link	+02:00	+03:00	IST
IM	Europe/Isle_of_Man		Link†	+00:00	+01:00	GMT
IN	Asia/Calcutta		Link	+05:30	+05:30	IST
IN	Asia/Kolkata		Canonical	+05:30	+05:30	IST
IO	Indian/Chagos		Canonical	+06:00	+06:00	6
IQ	Asia/Baghdad		Canonical	+03:00	+03:00	3
IR	Asia/Tehran		Canonical	+03:30	+03:30	330
IR	Iran		Link	+03:30	+03:30	330
IS	Atlantic/Reykjavik		Link†	+00:00	+00:00	GMT
IS	Iceland		Link	+00:00	+00:00	GMT
IT, SM, VA	Europe/Rome		Canonical	+01:00	+02:00	CET
JE	Europe/Jersey		Link†	+00:00	+01:00	GMT
JM	America/Jamaica		Canonical	−05:00	−05:00	EST
JM	Jamaica		Link	−05:00	−05:00	EST
JO	Asia/Amman		Canonical	+03:00	+03:00	3
JP	Asia/Tokyo		Canonical	+09:00	+09:00	JST
JP	Japan		Link	+09:00	+09:00	JST
KE, DJ, ER, ET, KM, MG, SO, TZ, UG, YT	Africa/Nairobi		Canonical	+03:00	+03:00	EAT
KG	Asia/Bishkek		Canonical	+06:00	+06:00	6
KH	Asia/Phnom_Penh		Link†	+07:00	+07:00	7
KI	Pacific/Enderbury		Link†	+13:00	+13:00	13
KI	Pacific/Kanton	Phoenix Islands	Canonical	+13:00	+13:00	13
KI	Pacific/Kiritimati	Line Islands	Canonical	+14:00	+14:00	14
KI, MH, TV, UM, WF	Pacific/Tarawa	Gilberts, Marshalls, Wake	Canonical	+12:00	+12:00	12
KM	Indian/Comoro		Link†	+03:00	+03:00	EAT
KN	America/St_Kitts		Link†	−04:00	−04:00	AST
KP	Asia/Pyongyang		Canonical	+09:00	+09:00	KST
KR	Asia/Seoul		Canonical	+09:00	+09:00	KST
KR	ROK		Link	+09:00	+09:00	KST
KW	Asia/Kuwait		Link†	+03:00	+03:00	3
KY	America/Cayman		Link†	−05:00	−05:00	EST
KZ	Asia/Almaty	most of Kazakhstan	Canonical	+06:00	+06:00	6
KZ	Asia/Aqtau	Mangghystaū/Mankistau	Canonical	+05:00	+05:00	5
KZ	Asia/Aqtobe	Aqtöbe/Aktobe	Canonical	+05:00	+05:00	5
KZ	Asia/Atyrau	Atyraū/Atirau/Gur'yev	Canonical	+05:00	+05:00	5
KZ	Asia/Oral	West Kazakhstan	Canonical	+05:00	+05:00	5
KZ	Asia/Qostanay	Qostanay/Kostanay/Kustanay	Canonical	+06:00	+06:00	6
KZ	Asia/Qyzylorda	Qyzylorda/Kyzylorda/Kzyl-Orda	Canonical	+05:00	+05:00	5
LA	Asia/Vientiane		Link†	+07:00	+07:00	7
LB	Asia/Beirut		Canonical	+02:00	+03:00	EET
LC	America/St_Lucia		Link†	−04:00	−04:00	AST
LI	Europe/Vaduz		Link†	+01:00	+02:00	CET
LK	Asia/Colombo		Canonical	+05:30	+05:30	530
LR	Africa/Monrovia		Canonical	+00:00	+00:00	GMT
LS	Africa/Maseru		Link†	+02:00	+02:00	SAST
LT	Europe/Vilnius		Canonical	+02:00	+03:00	EET
LU	Europe/Luxembourg		Link†	+01:00	+02:00	CET
LV	Europe/Riga		Canonical	+02:00	+03:00	EET
LY	Africa/Tripoli		Canonical	+02:00	+02:00	EET
LY	Libya		Link	+02:00	+02:00	EET
MA	Africa/Casablanca		Canonical	+01:00	+00:00	1
MC	Europe/Monaco		Link†	+01:00	+02:00	CET
MD	Europe/Chisinau		Canonical	+02:00	+03:00	EET
MD	Europe/Tiraspol		Link†	+02:00	+03:00	EET
ME	Europe/Podgorica		Link	+01:00	+02:00	CET
MF	America/Marigot		Link	−04:00	−04:00	AST
MG	Indian/Antananarivo		Link†	+03:00	+03:00	EAT
MH	Kwajalein		Link	+12:00	+12:00	12
MH	Pacific/Kwajalein	Kwajalein	Canonical	+12:00	+12:00	12
MH	Pacific/Majuro	most of Marshall Islands	Link†	+12:00	+12:00	12
MK	Europe/Skopje		Link†	+01:00	+02:00	CET
ML	Africa/Bamako		Link†	+00:00	+00:00	GMT
ML	Africa/Timbuktu		Link†	+00:00	+00:00	GMT
MM	Asia/Rangoon		Link	+06:30	+06:30	630
MM, CC	Asia/Yangon		Canonical	+06:30	+06:30	630
MN	Asia/Choibalsan	Dornod, Sükhbaatar	Canonical	+08:00	+08:00	8
MN	Asia/Hovd	Bayan-Ölgii, Govi-Altai, Hovd, Uvs, Zavkhan	Canonical	+07:00	+07:00	7
MN	Asia/Ulaanbaatar	most of Mongolia	Canonical	+08:00	+08:00	8
MN	Asia/Ulan_Bator		Link	+08:00	+08:00	8
MO	Asia/Macao		Link	+08:00	+08:00	CST
MO	Asia/Macau		Canonical	+08:00	+08:00	CST
MP	Pacific/Saipan		Link†	+10:00	+10:00	ChST
MQ	America/Martinique		Canonical	−04:00	−04:00	AST
MR	Africa/Nouakchott		Link†	+00:00	+00:00	GMT
MS	America/Montserrat		Link†	−04:00	−04:00	AST
MT	Europe/Malta		Canonical	+01:00	+02:00	CET
MU	Indian/Mauritius		Canonical	+04:00	+04:00	4
MV, TF	Indian/Maldives	Kerguelen, St Paul I, Amsterdam I	Canonical	+05:00	+05:00	5
MW	Africa/Blantyre		Link†	+02:00	+02:00	CAT
MX	America/Bahia_Banderas	Bahía de Banderas	Canonical	−06:00	−06:00	CST
MX	America/Cancun	Quintana Roo	Canonical	−05:00	−05:00	EST
MX	America/Chihuahua	Chihuahua (most areas)	Canonical	−06:00	−06:00	CST
MX	America/Ciudad_Juarez	Chihuahua (US border - west)	Canonical	−07:00	−06:00	MST
MX	America/Ensenada		Link†	−08:00	−07:00	PST
MX	America/Hermosillo	Sonora	Canonical	−07:00	−07:00	MST
MX	America/Matamoros	Coahuila, Nuevo León, Tamaulipas (US border)	Canonical	−06:00	−05:00	CST
MX	America/Mazatlan	Baja California Sur, Nayarit (most areas), Sinaloa	Canonical	−07:00	−07:00	MST
MX	America/Merida	Campeche, Yucatán	Canonical	−06:00	−06:00	CST
MX	America/Mexico_City	Central Mexico	Canonical	−06:00	−06:00	CST
MX	America/Monterrey	Durango; Coahuila, Nuevo León, Tamaulipas (most areas)	Canonical	−06:00	−06:00	CST
MX	America/Ojinaga	Chihuahua (US border - east)	Canonical	−06:00	−05:00	CST
MX	America/Santa_Isabel		Link	−08:00	−07:00	PST
MX	America/Tijuana	Baja California	Canonical	−08:00	−07:00	PST
MX	Mexico/BajaNorte		Link	−08:00	−07:00	PST
MX	Mexico/BajaSur		Link	−07:00	−07:00	MST
MX	Mexico/General		Link	−06:00	−06:00	CST
MY	Asia/Kuala_Lumpur	Malaysia (peninsula)	Link†	+08:00	+08:00	8
MY, BN	Asia/Kuching	Sabah, Sarawak	Canonical	+08:00	+08:00	8
MZ, BI, BW, CD, MW, RW, ZM, ZW	Africa/Maputo	Central Africa Time	Canonical	+02:00	+02:00	CAT
NA	Africa/Windhoek		Canonical	+02:00	+02:00	CAT
NC	Pacific/Noumea		Canonical	+11:00	+11:00	11
NE	Africa/Niamey		Link†	+01:00	+01:00	WAT
NF	Pacific/Norfolk		Canonical	+11:00	+12:00	11
NG, AO, BJ, CD, CF, CG, CM, GA, GQ, NE	Africa/Lagos	West Africa Time	Canonical	+01:00	+01:00	WAT
NI	America/Managua		Canonical	−06:00	−06:00	CST
NL	Europe/Amsterdam		Link†	+01:00	+02:00	CET
NO	Europe/Oslo		Link†	+01:00	+02:00	CET
NP	Asia/Kathmandu		Canonical	+05:45	+05:45	545
NP	Asia/Katmandu		Link	+05:45	+05:45	545
NR	Pacific/Nauru		Canonical	+12:00	+12:00	12
NU	Pacific/Niue		Canonical	−11:00	−11:00	-11
NZ	NZ		Link	+12:00	+13:00	NZST
NZ	NZ-CHAT		Link	+12:45	+13:45	1245
NZ, AQ	Pacific/Auckland	New Zealand time	Canonical	+12:00	+13:00	NZST
NZ	Pacific/Chatham	Chatham Islands	Canonical	+12:45	+13:45	1245
OM	Asia/Muscat		Link†	+04:00	+04:00	4
PA, CA, KY	America/Panama	EST - ON (Atikokan), NU (Coral H)	Canonical	−05:00	−05:00	EST
PE	America/Lima		Canonical	−05:00	−05:00	-5
PF	Pacific/Gambier	Gambier Islands	Canonical	−09:00	−09:00	-9
PF	Pacific/Marquesas	Marquesas Islands	Canonical	−09:30	−09:30	-930
PF	Pacific/Tahiti	Society Islands	Canonical	−10:00	−10:00	-10
PG	Pacific/Bougainville	Bougainville	Canonical	+11:00	+11:00	11
PG, AQ, FM	Pacific/Port_Moresby	Papua New Guinea (most areas), Chuuk, Yap, Dumont d'Urville	Canonical	+10:00	+10:00	10
PH	Asia/Manila		Canonical	+08:00	+08:00	PHT
PK	Asia/Karachi		Canonical	+05:00	+05:00	PKT
PL	Europe/Warsaw		Canonical	+01:00	+02:00	CET
PL	Poland		Link	+01:00	+02:00	CET
PM	America/Miquelon		Canonical	−03:00	−02:00	-3
PN	Pacific/Pitcairn		Canonical	−08:00	−08:00	-8
PR, AG, CA, AI, AW, BL, BQ, CW, DM, GD, GP, KN, LC, MF, MS, SX, TT, VC, VG, VI	America/Puerto_Rico	AST	Canonical	−04:00	−04:00	AST
PS	Asia/Gaza	Gaza Strip	Canonical	+02:00	+03:00	EET
PS	Asia/Hebron	West Bank	Canonical	+02:00	+03:00	EET
PT	Atlantic/Azores	Azores	Canonical	−01:00	+00:00	-1
PT	Atlantic/Madeira	Madeira Islands	Canonical	+00:00	+01:00	WET
PT	Europe/Lisbon	Portugal (mainland)	Canonical	+00:00	+01:00	WET
PT	Portugal		Link	+00:00	+01:00	WET
PW	Pacific/Palau		Canonical	+09:00	+09:00	9
PY	America/Asuncion		Canonical	−04:00	−03:00	-4
QA, BH	Asia/Qatar		Canonical	+03:00	+03:00	3
RE	Indian/Reunion		Link†	+04:00	+04:00	4
RO	Europe/Bucharest		Canonical	+02:00	+03:00	EET
RS, BA, HR, ME, MK, SI	Europe/Belgrade		Canonical	+01:00	+02:00	CET
RU	Asia/Anadyr	MSK+09 - Bering Sea	Canonical	+12:00	+12:00	12
RU	Asia/Barnaul	MSK+04 - Altai	Canonical	+07:00	+07:00	7
RU	Asia/Chita	MSK+06 - Zabaykalsky	Canonical	+09:00	+09:00	9
RU	Asia/Irkutsk	MSK+05 - Irkutsk, Buryatia	Canonical	+08:00	+08:00	8
RU	Asia/Kamchatka	MSK+09 - Kamchatka	Canonical	+12:00	+12:00	12
RU	Asia/Khandyga	MSK+06 - Tomponsky, Ust-Maysky	Canonical	+09:00	+09:00	9
RU	Asia/Krasnoyarsk	MSK+04 - Krasnoyarsk area	Canonical	+07:00	+07:00	7
RU	Asia/Magadan	MSK+08 - Magadan	Canonical	+11:00	+11:00	11
RU	Asia/Novokuznetsk	MSK+04 - Kemerovo	Canonical	+07:00	+07:00	7
RU	Asia/Novosibirsk	MSK+04 - Novosibirsk	Canonical	+07:00	+07:00	7
RU	Asia/Omsk	MSK+03 - Omsk	Canonical	+06:00	+06:00	6
RU	Asia/Sakhalin	MSK+08 - Sakhalin Island	Canonical	+11:00	+11:00	11
RU	Asia/Srednekolymsk	MSK+08 - Sakha (E); N Kuril Is	Canonical	+11:00	+11:00	11
RU	Asia/Tomsk	MSK+04 - Tomsk	Canonical	+07:00	+07:00	7
RU	Asia/Ust-Nera	MSK+07 - Oymyakonsky	Canonical	+10:00	+10:00	10
RU	Asia/Vladivostok	MSK+07 - Amur River	Canonical	+10:00	+10:00	10
RU	Asia/Yakutsk	MSK+06 - Lena River	Canonical	+09:00	+09:00	9
RU	Asia/Yekaterinburg	MSK+02 - Urals	Canonical	+05:00	+05:00	5
RU	Europe/Astrakhan	MSK+01 - Astrakhan	Canonical	+04:00	+04:00	4
RU	Europe/Kaliningrad	MSK-01 - Kaliningrad	Canonical	+02:00	+02:00	EET
RU	Europe/Kirov	MSK+00 - Kirov	Canonical	+03:00	+03:00	MSK
RU	Europe/Moscow	MSK+00 - Moscow area	Canonical	+03:00	+03:00	MSK
RU	Europe/Samara	MSK+01 - Samara, Udmurtia	Canonical	+04:00	+04:00	4
RU	Europe/Saratov	MSK+01 - Saratov	Canonical	+04:00	+04:00	4
RU, UA	Europe/Simferopol	Crimea	Canonical	+03:00	+03:00	MSK
RU	Europe/Ulyanovsk	MSK+01 - Ulyanovsk	Canonical	+04:00	+04:00	4
RU	Europe/Volgograd	MSK+00 - Volgograd	Canonical	+03:00	+03:00	MSK
RU	W-SU		Link	+03:00	+03:00	MSK
RW	Africa/Kigali		Link†	+02:00	+02:00	CAT
SA, AQ, KW, YE	Asia/Riyadh	Syowa	Canonical	+03:00	+03:00	3
SB, FM	Pacific/Guadalcanal	Pohnpei	Canonical	+11:00	+11:00	11
SC	Indian/Mahe		Link†	+04:00	+04:00	4
SD	Africa/Khartoum		Canonical	+02:00	+02:00	CAT
SE	Europe/Stockholm		Link†	+01:00	+02:00	CET
SG, MY	Asia/Singapore	peninsular Malaysia	Canonical	+08:00	+08:00	8
SG	Singapore		Link	+08:00	+08:00	8
SH	Atlantic/St_Helena		Link†	+00:00	+00:00	GMT
SI	Europe/Ljubljana		Link†	+01:00	+02:00	CET
SJ	Arctic/Longyearbyen		Link	+01:00	+02:00	CET
SJ	Atlantic/Jan_Mayen		Link†	+01:00	+02:00	CET
SK	Europe/Bratislava		Link	+01:00	+02:00	CET
SL	Africa/Freetown		Link†	+00:00	+00:00	GMT
SM	Europe/San_Marino		Link	+01:00	+02:00	CET
SN	Africa/Dakar		Link†	+00:00	+00:00	GMT
SO	Africa/Mogadishu		Link†	+03:00	+03:00	EAT
SR	America/Paramaribo		Canonical	−03:00	−03:00	-3
SS	Africa/Juba		Canonical	+02:00	+02:00	CAT
ST	Africa/Sao_Tome		Canonical	+00:00	+00:00	GMT
SV	America/El_Salvador		Canonical	−06:00	−06:00	CST
SX	America/Lower_Princes		Link	−04:00	−04:00	AST
SY	Asia/Damascus		Canonical	+03:00	+03:00	3
SZ	Africa/Mbabane		Link†	+02:00	+02:00	SAST
TC	America/Grand_Turk		Canonical	−05:00	−04:00	EST
TD	Africa/Ndjamena		Canonical	+01:00	+01:00	WAT
TF	Indian/Kerguelen		Link†	+05:00	+05:00	5
TG	Africa/Lome		Link†	+00:00	+00:00	GMT
TH, CX, KH, LA, VN	Asia/Bangkok	north Vietnam	Canonical	+07:00	+07:00	7
TJ	Asia/Dushanbe		Canonical	+05:00	+05:00	5
TK	Pacific/Fakaofo		Canonical	+13:00	+13:00	13
TL	Asia/Dili		Canonical	+09:00	+09:00	9
TM	Asia/Ashgabat		Canonical	+05:00	+05:00	5
TM	Asia/Ashkhabad		Link	+05:00	+05:00	5
TN	Africa/Tunis		Canonical	+01:00	+01:00	CET
TO	Pacific/Tongatapu		Canonical	+13:00	+13:00	13
TR	Asia/Istanbul		Link	+03:00	+03:00	3
TR	Europe/Istanbul		Canonical	+03:00	+03:00	3
TR	Turkey		Link	+03:00	+03:00	3
TT	America/Port_of_Spain		Link†	−04:00	−04:00	AST
TV	Pacific/Funafuti		Link†	+12:00	+12:00	12
TW	Asia/Taipei		Canonical	+08:00	+08:00	CST
TW	ROC		Link	+08:00	+08:00	CST
TZ	Africa/Dar_es_Salaam		Link†	+03:00	+03:00	EAT
UA	Europe/Kiev		Link	+02:00	+03:00	EET
UA	Europe/Kyiv	most of Ukraine	Canonical	+02:00	+03:00	EET
UA	Europe/Uzhgorod		Link†	+02:00	+03:00	EET
UA	Europe/Zaporozhye		Link†	+02:00	+03:00	EET
UG	Africa/Kampala		Link†	+03:00	+03:00	EAT
UM	Pacific/Midway	Midway Islands	Link†	−11:00	−11:00	SST
UM	Pacific/Wake	Wake Island	Link†	+12:00	+12:00	12
US	America/Adak	Alaska - western Aleutians	Canonical	−10:00	−09:00	HST
US	America/Anchorage	Alaska (most areas)	Canonical	−09:00	−08:00	AKST
US	America/Atka		Link	−10:00	−09:00	HST
US	America/Boise	Mountain - ID (south); OR (east)	Canonical	−07:00	−06:00	MST
US	America/Chicago	Central (most areas)	Canonical	−06:00	−05:00	CST
US	America/Denver	Mountain (most areas)	Canonical	−07:00	−06:00	MST
US	America/Detroit	Eastern - MI (most areas)	Canonical	−05:00	−04:00	EST
US	America/Fort_Wayne		Link	−05:00	−04:00	EST
US	America/Indiana/Indianapolis	Eastern - IN (most areas)	Canonical	−05:00	−04:00	EST
US	America/Indiana/Knox	Central - IN (Starke)	Canonical	−06:00	−05:00	CST
US	America/Indiana/Marengo	Eastern - IN (Crawford)	Canonical	−05:00	−04:00	EST
US	America/Indiana/Petersburg	Eastern - IN (Pike)	Canonical	−05:00	−04:00	EST
US	America/Indiana/Tell_City	Central - IN (Perry)	Canonical	−06:00	−05:00	CST
US	America/Indiana/Vevay	Eastern - IN (Switzerland)	Canonical	−05:00	−04:00	EST
US	America/Indiana/Vincennes	Eastern - IN (Da, Du, K, Mn)	Canonical	−05:00	−04:00	EST
US	America/Indiana/Winamac	Eastern - IN (Pulaski)	Canonical	−05:00	−04:00	EST
US	America/Indianapolis		Link	−05:00	−04:00	EST
US	America/Juneau	Alaska - Juneau area	Canonical	−09:00	−08:00	AKST
US	America/Kentucky/Louisville	Eastern - KY (Louisville area)	Canonical	−05:00	−04:00	EST
US	America/Kentucky/Monticello	Eastern - KY (Wayne)	Canonical	−05:00	−04:00	EST
US	America/Knox_IN		Link	−06:00	−05:00	CST
US	America/Los_Angeles	Pacific	Canonical	−08:00	−07:00	PST
US	America/Louisville		Link	−05:00	−04:00	EST
US	America/Menominee	Central - MI (Wisconsin border)	Canonical	−06:00	−05:00	CST
US	America/Metlakatla	Alaska - Annette Island	Canonical	−09:00	−08:00	AKST
US	America/New_York	Eastern (most areas)	Canonical	−05:00	−04:00	EST
US	America/Nome	Alaska (west)	Canonical	−09:00	−08:00	AKST
US	America/North_Dakota/Beulah	Central - ND (Mercer)	Canonical	−06:00	−05:00	CST
US	America/North_Dakota/Center	Central - ND (Oliver)	Canonical	−06:00	−05:00	CST
US	America/North_Dakota/New_Salem	Central - ND (Morton rural)	Canonical	−06:00	−05:00	CST
US, CA	America/Phoenix	MST - AZ (most areas), Creston BC	Canonical	−07:00	−07:00	MST
US	America/Shiprock		Link	−07:00	−06:00	MST
US	America/Sitka	Alaska - Sitka area	Canonical	−09:00	−08:00	AKST
US	America/Yakutat	Alaska - Yakutat	Canonical	−09:00	−08:00	AKST
US	Navajo		Link	−07:00	−06:00	MST
US	Pacific/Honolulu	Hawaii	Canonical	−10:00	−10:00	HST
US	Pacific/Johnston		Link†	−10:00	−10:00	HST
US	US/Alaska		Link	−09:00	−08:00	AKST
US	US/Aleutian		Link	−10:00	−09:00	HST
US	US/Arizona		Link	−07:00	−07:00	MST
US	US/Central		Link	−06:00	−05:00	CST
US	US/East-Indiana		Link	−05:00	−04:00	EST
US	US/Eastern		Link	−05:00	−04:00	EST
US	US/Hawaii		Link	−10:00	−10:00	HST
US	US/Indiana-Starke		Link	−06:00	−05:00	CST
US	US/Michigan		Link	−05:00	−04:00	EST
US	US/Mountain		Link	−07:00	−06:00	MST
US	US/Pacific		Link	−08:00	−07:00	PST
UY	America/Montevideo		Canonical	−03:00	−03:00	-3
UZ	Asia/Samarkand	Uzbekistan (west)	Canonical	+05:00	+05:00	5
UZ	Asia/Tashkent	Uzbekistan (east)	Canonical	+05:00	+05:00	5
VA	Europe/Vatican		Link	+01:00	+02:00	CET
VC	America/St_Vincent		Link†	−04:00	−04:00	AST
VE	America/Caracas		Canonical	−04:00	−04:00	-4
VG	America/Tortola		Link†	−04:00	−04:00	AST
VI	America/St_Thomas		Link†	−04:00	−04:00	AST
VI	America/Virgin		Link	−04:00	−04:00	AST
VN	Asia/Ho_Chi_Minh	south Vietnam	Canonical	+07:00	+07:00	7
VN	Asia/Saigon		Link	+07:00	+07:00	7
VU	Pacific/Efate		Canonical	+11:00	+11:00	11
WF	Pacific/Wallis		Link†	+12:00	+12:00	12
WS	Pacific/Apia		Canonical	+13:00	+13:00	13
YE	Asia/Aden		Link†	+03:00	+03:00	3
YT	Indian/Mayotte		Link†	+03:00	+03:00	EAT
ZA, LS, SZ	Africa/Johannesburg		Canonical	+02:00	+02:00	SAST
ZM	Africa/Lusaka		Link†	+02:00	+02:00	CAT
ZW	Africa/Harare		Link†	+02:00	+02:00	CAT'''


class Timezone(Expression):
    def __init__(self):
        super().__init__()
        self.fn = Function(FUNCTION_DESCRIPTION)

    def find_substring(self, country):
        zones = COUNTRY_TIME_TABLE.split('\n')
        zones = [zone.lower() for zone in zones]
        similarities = {zone: SequenceMatcher(None, country.lower(), zone).ratio() for zone in zones}
        return similarities

    def forward(self, request, k: int = 10, *args, **kwargs):
        request       = self._to_symbol(request)
        country       = request.extract("extract the target timezone or country")
        similarities  = self.find_substring(str(request))
        # get top k indexes
        top_k_zones = sorted(similarities, key=similarities.get, reverse=True)[:k]
        data = f"Timezones: {top_k_zones}\n{request}"
        return self.fn(data, *args, **kwargs)
