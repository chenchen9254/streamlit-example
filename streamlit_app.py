import spacy
import spacy_streamlit
import legalSentencizer
import nltk
import somajo
from spacy_streamlit import visualize_ner
import jsonlines
import random

def select_data(n):
    data = []
    with jsonlines.open('data/Einkommensteuer.jsonl', 'r') as f:
        for line in f:
            data.append(line)
    data_n = random.sample(data, n)
    text = []
    for line in data_n:
        (key, value), = line.items()
        text.append(value)
    return text

source_nlp = spacy.load("de_core_news_md")
MODEL_PATH = ["output/model-best"]
nlp = spacy.load(MODEL_PATH[0])
nlp.add_pipe("parser", source=source_nlp, before='ner')
legalSentencizer.add_to_pipe(nlp)
visualizer = ['ner']
#text = "Ausscheiden aus Gesellschaft - Vorsteuerbelastete Abwicklungskosten (\u00a7 15 Abs. 1 UStG) Nach Ansicht des Autors stellt die zit. Entscheidung des BFH vom 20. 7. 1988 die konsequente Fortf\u00fchrung der verfehlten Rechtsprechung zum Vorsteuerabzug von Gesellschaftern dar. Er bem\u00e4ngelt auch an diesem Urteil, da\u00df allein auf den Wortlaut des Gesetzes abgestellt und nicht gefragt werde, ob das gefundene Ergebnis auch dem Gesetzesziel entspricht. Das Gesetzesziel verlange den Vorsteuerabzug bei allen Aufwendungen, die mit der unternehmerischen Bet\u00e4tigung zusammenh\u00e4ngen. Bei wem der Vorsteuerabzug in Betracht kommt, h\u00e4nge davon ab, wer die Aufwendungen getragen hat. \u00a7 15 UStG k\u00f6nne in diesem Sinne ausgelegt werden, indem dem Gesellschafter f\u00fcr den Vorsteuerabzug insoweit die Unternehmereigenschaft der Gesellschaft zugerechnet werde. 0093121 Anmerkung Umsatzsteuer Stadie Prof. Dr. Holger 08.09.1989 19890908 UR-1989-0278 UR-1989-0279 UStG:15/1 1989 BFH Urteil X R 46/81 v. 20. 7. 1988 BFH/NV 1989 326"
#doc = nlp(text)
texts = ['Schon aus diesem Grund könne das Urteil des Bundesgerichtshofs niemals Rechts- und Billigkeitsgrundlage für den am 30. Dezember 1994 abgeschlossenen Vergleich sein.',
         'Der räumliche Zusammenhang ist unschädlich, wenn die andere Wohnung unentgeltlich einem Angehörigen überlassen wird (BFH vom 28.6.2002-BStBl 2003 II S. 119) ; wird die zweite Wohnung jedoch von einem minderjährigen Kind der Familie bewohnt, ist die Förderung für diese Wohnung ausgeschlossen, weil der elterliche Haushalt die Wohnung des minderjährigen Kindes umfasst.',
         'Sie beantragten die Erhöhung des Verlustes gemäß § 17 EStG aufgrund von Bürgschaftsinanspruchnahmen.'
         'In diesem Fall sind für die dann erforderliche Ertragsprognose die Verhältnisse ab dem Zeitpunkt der Teilübertragung maßgeblich (vgl BFH v 17.03.2010, BStBl II 2011, 622) .',
         '115 Die aufgezeigte Verzinsung kann nicht durch freiwillige Auflösung im Laufe des Wj verhindert werden (BFH v 26.10.89, IV R 83/88, BStBl II 1990, 290) .',
         'Bei gewichtigen Fällen empfiehlt sich die Einholung einer verbindlichen Auskunft nach § 89 AO.'
         'Die Möglichkeit, dass das Gebäude für Zwecke der eigenen Vermögensverwaltung hergestellt wird, scheidet dann aus (vgl. BFH-Urteil vom 12. Juli 2007 X R 4/04, BStBl II 2007, 885) .'
         'Es werden bei der Einräumung des Optionsrechts durch Abschluss des Stillhaltervertrags und bei der Übertragung des Wirtschaftsguts durch Abschluss eines Veräußerungsvertrags identische Leistungen erbracht (Philipowski, Vereinnahmte Stillhalterprämien; Gezahlter Barausgleich nicht abziehbar?',
         '6b Abs 2a S 4-6 EStG nF: § 6b Abs 2a EStG enthält ein Wahlrecht, dass die auf den nach § 6b EStG begünstigten Veräußerungsgewinn entfallende festgesetzte Steuer zinslos in fünf gleichen Jahresraten gezahlt werden kann, wenn der StPfl eine Reinvestition in BV im EU- bzw EWR-Ausland plant.',
         'Weder das vom Kläger in Bezug genommene Urteil des BFH vom 27.5.2003 (VI R 33/01) als solches noch die darin zum Ausdruck gebrachte Auffassung, Aufwendungen für die erstmalige Berufsausbildung seien unter bestimmten Voraussetzungen als-vorweggenommene-Werbungskosten zu berücksichtigen, rechtfertigen eine Änderung der bestandskräftigen Einkommensteuerbescheide für die Streitjahre nach § 173 Abs. 1 Nr. 2 oder § 175 Abs. 1 Nr. 2 AO.']

data_50 = select_data(50)
docs = list(nlp.pipe(data_50))
#spacy_streamlit.visualize(MODEL_PATH, text, visualizer)
color_mapping = {}
entities_name = ['GS', 'VS', 'RS', 'UN', 'PER', 'LIT', 'VT', 'GRT', 'INN', 'EUN',
                 'LDS', 'ORG', 'LD', 'ST', 'STR', 'VO', 'AN', 'RR', 'MRK']
hex_values = ["#e6194B", "#fabed4", "#ffd8b1", "#f58231", "#800000", "#9A6324", "#808000", "#ffe119", "#fffac8",
              "#bfef45", "#3cb44b", "#aaffc3", "#42d4f4", "#469990", "#4363d8", "#911eb4", "#dcbeff", "#f032e6"]

for entity, hex_value in zip(entities_name, hex_values):
        color_mapping[entity] = hex_value

for i in range(len(docs)):
    visualize_ner(docs[i], labels=nlp.get_pipe("ner").labels, key=i, colors=color_mapping)
