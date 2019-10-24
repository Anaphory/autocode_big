import sys
import itertools

from tqdm import tqdm

import sqlalchemy

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship, aliased, backref
from sqlalchemy import create_engine

if sys.argv[1]:
    engine = create_engine(sys.argv[1], echo=True)
    # For SQLite:
    engine.execute('PRAGMA FOREIGN_KEYS=ON;')
else:
    engine = create_engine(
        'postgresql://lexirumah:lexirumah@localhost:5432/lexirumah', echo=True)

Base = declarative_base()

class Language(Base):
    __tablename__ = 'language'

    id = Column(String, primary_key=True)
    language = Column(String)

    def __repr__(self):
        return self.id

class Scorer(Base):
    __tablename__ = 'scorers'
    id = Column(Integer, primary_key=True)
    language1_id = Column(String, ForeignKey(Language.id, ondelete="CASCADE"))
    language2_id = Column(String, ForeignKey(Language.id, ondelete="CASCADE"))
    language1 = relationship(
        Language, foreign_keys=language1_id,
        backref=backref("scorers1"))
    language2 = relationship(
        Language, foreign_keys=language2_id,
        backref=backref("scorers2"))
    scorer = Column(String)

class Concept(Base):
    __tablename__ = 'concept'

    id = Column(String, primary_key=True)
    concept = Column(String)

    def __repr__(self):
        return self.id

class Form(Base):
    __tablename__ = 'form'

    id = Column(String, primary_key=True)
    language_id = Column(String, ForeignKey(Language.id, ondelete="CASCADE"), index=True)
    language = relationship(
        Language,
        backref=backref("forms"))
    concept_id = Column(String, ForeignKey(Concept.id, ondelete="CASCADE"))
    concept = relationship(
        Concept,
        backref=backref("forms"))
    transcription = Column(String)
    soundclasses = Column(String)
    alignment = Column(String)
    connected_component = Column(String)
    cognateset = Column(String)

    def __repr__(self):
        return "<Form {:}: {:} in {:} is [{:}]>".format(
            self.id, self.concept, self.language, self.transcription)

F = aliased(Form)

class Similarity(Base):
    __tablename__ = 'similarities'
    id = Column(Integer, primary_key=True)
    form1_id = Column(String, ForeignKey(Form.id, ondelete="CASCADE"), index=True)
    form2_id = Column(String, ForeignKey(Form.id, ondelete="CASCADE"), index=True)
    form1 = relationship(
        Form, foreign_keys=form1_id,
        backref=backref("similarities1"))
    form2 = relationship(
        Form, foreign_keys=form2_id,
        backref=backref("similarities2"))
    score = Column(Float)

    def __repr__(self):
        return "{:}/{:}:{:}".format(self.form1, self.form2, self.score)

Base.metadata.create_all(engine)

from sqlalchemy.orm import sessionmaker
Session = sessionmaker(bind=engine)
Session.configure(bind=engine)  # once engine is available
session = Session()

import pylexirumah

dataset = pylexirumah.get_dataset()
from lingpy.convert.strings import scorer2str
from lingpy.read.qlc import read_scorer
from lingpy.compare.partial import Partial
import lingpy


def import_all_languages():
    for language in dataset["LanguageTable"].iterdicts():
        session.add(Language(
            id=language["ID"],
            language=language["Name"]))
        try:
            session.commit()
        except sqlalchemy.exc.IntegrityError:
            print("Language {:} already exists.".format(language["ID"]))
            session.rollback()

def import_all_parameters():
    for concept in dataset["ParameterTable"].iterdicts():
        session.add(Concept(
            id=concept["ID"],
            concept=concept["English"]))
    session.commit()

def import_all_forms(filter=None):
    for form in dataset["FormTable"].iterdicts():
        if filter is None or filter(form):
            session.add(Form(
                id=form["ID"],
                language_id=form["Lect_ID"],
                concept_id=form["Concept_ID"],
                transcription=form["Form"],
                soundclasses=" ".join(form["Segments"])))
    session.commit()


def calculate_scores(*languages):
    pair_of_languages = {
        0: ["backreference", "doculect", "concept", "ipa", "tokens"]
    }
    for l in languages:
        pair_of_languages.update({
            i + len(pair_of_languages):
            [form.id,
            form.language_id,
            form.concept_id,
            form.transcription,
            form.soundclasses.split()]
            for i, form in enumerate(session.query(Form).filter_by(language=l))
        })

    lex = Partial(pair_of_languages, model=lingpy.data.model.Model("asjp"),
                  check=True, apply_checks=True)
    lex.get_scorer(runs=10000, ratio=(3, 1), threshold=0.7)

    # This does not generalize to non-two languages yet
    session.add(Scorer(language1=languages[0],
                       language2=languages[1],
                       scorer=scorer2str(lex.bscorer)))

    for concept, forms, matrix in lex._get_matrices(
            method='lexstat',
            scale=0.5,
            factor=0.3,
            restricted_chars="_T",
            mode="overlap",
            gop=-2,
            restriction=""):
        for (i1, f1), (i2, f2) in itertools.combinations(enumerate(forms), 2):
            f1 = lex[f1][0] # Index 0 contains the 'backref', ie. our ID
            f2 = lex[f2][0] # Index 0 contains the 'backref', ie. our ID
            session.add(Similarity(
                form1_id=f1,
                form2_id=f2,
                score=matrix[i1][i2]))

    session.commit()

def calculate_all_scores():
    session.query(Similarity).delete()
    session.commit()
    for l1, l2 in tqdm(itertools.combinations(session.query(Language), r=2)):
        calculate_scores(l1, l2)

concept_numbers = {}

def find_connected_components(threshold=0.55, reset=False, source="biglexstat"):
    if threshold > 2:
        for form in session.query(Form):
            form.connected_component = form.concept
        session.commit()
        return
    if reset:
        for form in session.query(Form):
            form.connected_component = None
        session.commit()

    forms = session.query(Form, F).join(
        Similarity, Form.id==Similarity.form1_id).join(
            F, F.id==Similarity.form2_id).filter(
                Similarity.score < threshold,
                Form.connected_component==None,
                F.connected_component==None).first()

    components = set()
    while forms:
        f1, f2 = forms
        component_index = 1
        while "{:}-{:}".format(f1.concept, component_index) in components:
            component_index += 1
        print(f1.concept, component_index, f1, f2)
        component_index = "{:}-{:}".format(f1.concept, component_index)
        f1.connected_component = component_index
        f2.connected_component = component_index
        session.commit()

        added = True
        while added:
            added = False
            for form in session.query(Form).join(
                    Similarity, Form.id==Similarity.form2_id).join(
                        F, F.id==Similarity.form1_id).filter(
                            Similarity.score < threshold,
                            F.connected_component == component_index,
                            Form.connected_component == None):
                form.connected_component = component_index
                components.add(component_index)
                added = True
            session.commit()

            for form in session.query(Form).join(
                    Similarity, Form.id==Similarity.form1_id).join(
                        F, F.id==Similarity.form2_id).filter(
                            Similarity.score < threshold,
                            F.connected_component == component_index,
                            Form.connected_component == None):
                form.connected_component = component_index
                components.add(component_index)
                added = True
            session.commit()

        session.commit()
        forms = session.query(Form, F).join(
            Similarity, Form.id==Similarity.form1_id).join(
                F, F.id==Similarity.form2_id).filter(
                    Similarity.score < threshold,
                    Form.connected_component==None,
                    F.connected_component==None).first()

    for f1 in session.query(Form).filter(Form.connected_component==None):
        component_index = 1
        while "{:}-{:}".format(f1.concept, component_index) in components:
            component_index += 1
        component_index = "{:}-{:}".format(f1.concept, component_index)
        form.connected_component = component_index
        components.add(component_index)
    session.commit()

    return components

import igraph as ig
import networkx

def cluster_connected_component(component, threshold=0.55):
    for form in session.query(Form).filter_by(connected_component=component):
        form.cognateset = None
    session.commit()

    g = ig.Graph()

    vertices = dict(session.query(Form.id, Form).filter(
            Form.connected_component == component).all())
    v = list(vertices)
    g.add_vertices(v)

    F = aliased(Form)

    edges = list(session.query(Form.id).join(
        Similarity, Form.id==Similarity.form1_id).join(
            F, F.id==Similarity.form2_id).add_columns(
                F.id).filter(
                    Form.connected_component == component,
                    F.connected_component == component,
                    Similarity.score < threshold
                ).all())
    g.add_edges(edges)

    components = g.community_infomap(edge_weights=None,
                                     vertex_weights=None)
    for s, subgraph in enumerate(components.subgraphs()):
        community_id = "{:}-{:}".format(component, s)
        for vertex in subgraph.vs:
            form = session.query(Form).filter_by(id=vertex["name"]).one()
            form.cognateset = community_id
        session.commit()

def cluster_all_connected_components(threshold=0.55):
    for component, in set(session.query(Form.connected_component).all()):
        if component:
            cluster_connected_component(component, threshold=threshold)

def add_language(language_id, threshold=0.55):
    language = session.query(Language).filter_by(id=language_id).first()
    if language:
        session.delete(language)
        session.commit()

        print("Deleted previous data about {:}".format(language_id))

    for language in dataset["LanguageTable"].iterdicts():
        if language["ID"] == language_id:
            lg = Language(
                id=language["ID"],
                language=language["Name"])
            session.add(lg)
    session.commit()

    for form in dataset["FormTable"].iterdicts():
        if form["Lect_ID"] == language_id:
            session.add(Form(
                id=form["ID"],
                language_id=form["Lect_ID"],
                concept_id=form["Concept_ID"],
                transcription=form["Form"],
                soundclasses=" ".join(form["Segments"])))
    session.commit()

    print("Imported data about {:}".format(language_id))

    for l2 in tqdm(session.query(Language)):
        try:
            calculate_scores(lg, l2)
        except ValueError:
            continue

    reset_components = set()
    for c1, c2 in session.query(F.connected_component, Form.connected_component).join(
            Similarity, Form.id==Similarity.form1_id).join(
                    F, F.id==Similarity.form2_id).filter(
                    Similarity.score < threshold,
                    (F.language_id == language_id) | (Form.language_id == language_id)):
        reset_components.add(c1)
        reset_components.add(c2)
    reset_components.remove(None)

    for component in reset_components:
        for form in session.query(Form).filter(Form.connected_component == component):
            form.connected_component = None
    session.commit()

    for component in find_connected_components():
        cluster_connected_component(component)

def create_cognateset_table(source="biglexstat"):
    dataset["CognateTable"].write([
        {
            "ID": "{:}-{:}".format(form.id, source),
            "Form_ID": form.id,
            "Cognateset_ID": form.cognateset,
            "Alignment": form.alignment,
            "Source": [source]
        }
        for form in session.query(Form).filter(Form.cognateset != None)
    ])

def cognate_code(threshold=0.55, start=None):
    if start is None:
        calculate_all_scores()
    else:
        code = False
        for l1, l2 in tqdm(itertools.combinations(session.query(Language), r=2)):
            if l1.id == start:
                code = True
            if code:
                calculate_scores(l1, l2)
    find_connected_components(threshold=threshold, reset=True)
    cluster_all_connected_components(threshold=threshold)
    create_cognateset_table()


def connections_in_connected_component(component_id):
    return session.query(Form, F, Similarity.score).join(
        Similarity, Form.id==Similarity.form1_id).join(
            F, F.id==Similarity.form2_id).filter(
                Form.connected_component == component_id,
                F.connected_component == component_id,
                Similarity.score < 0.55)

import matplotlib
import matplotlib.pyplot as plt
import networkx
def plot(edges):
    graph = networkx.Graph()
    graph.add_weighted_edges_from((f1, f2, 1-d) for f1, f2, d in edges)
    layout = networkx.spring_layout(graph, iterations=200)
    colors = {}
    for n, node in enumerate(graph):
        colors[node.cognateset] = len(colors)
    networkx.draw(graph,
        pos=layout,
        node_color=[colors[node.cognateset] for node in graph],
        cmap=matplotlib.cm.jet,
        width=[d['weight'] * 5 for n1, n2, d in graph.edges(data=True)],
        edge_color=[1 - d['weight'] for n1, n2, d in graph.edges(data=True)],
        edge_cmap=matplotlib.cm.magma,
        edge_vmin=0, edge_vmax=1,
        )
    networkx.draw_networkx_labels(
            graph,
            pos=layout,
            labels={n: "{:}\n{:}".format(n.language_id, n.transcription) for n in graph})
    plt.show()
    return graph


class VerboseDict(dict):
    def __getitem__(self, key):
        print(key)
        return super(self).__getitem__(key)


def scorer(l1, l2):
    return read_scorer(
        session.query(Scorer.scorer).filter(Scorer.language1_id == l1, Scorer.language2_id == l2).one()
    [0])


def align(cognateset):
    forms = session.query(Form, Form.soundclasses).filter(
        Form.cognateset == cognateset)
    languages = session.query(Form.language_id).filter(Form.cognateset == cognateset).distinct()
    # scoredict = VerboseDict()
    # for (l1,), (l2,) in itertools.combinations_with_replacement(languages, 2):
    #     try:
    #         scoredict[l1, l2] = scorer(l1, l2)
    #     except sqlalchemy.orm.exc.NoResultFound:
    #         pass
    sounds = [classes.split() for form, classes in forms]
    if len(sounds) < 2:
        for form, classes in forms:
            form.alignment = " ".join(classes.split())
        return

    m = lingpy.align.multiple.Multiple(sounds)
    m.prog_align()
    for (form, classes), alignment in zip(forms, m.alm_matrix):
        form.alignment = " ".join(alignment)

def align_all():
    cognatesets = session.query(Form.cognateset).distinct()
    for cognateset in cognatesets:
        try:
            align(cognateset)
        except IndexError:
            continue

import matplotlib

def cognate_code_and_align(threshold=0.55, start=None):
    cognate_code(threshold, start)
    align_all()
    create_cognateset_table()

