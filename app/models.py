from . import db


class TrainingSet(db.Model):

    __tablename__ = 'traingset'
    id = db.Column(db.Integer, primary_key=True)
    db_type = db.Column(db.String(64))
    classtag = db.Column(db.String(64))
    text = db.Column(db.Text())
    is_cut = db.Column(db.Boolean, default=False)
    cut_text = db.Column(db.Text())

