import os

from flask import Blueprint, abort, redirect, render_template, request, session, url_for

from ice_servers import get_ice_servers
from tutormatch import (
    GRADE_LEVELS,
    QUIZ_QUESTIONS,
    SUBJECTS,
    TEACHING_QUESTIONS,
    attach_participant,
    build_profile_sentence,
    clean_answers,
    describe_learning_style,
    describe_teaching_style,
    get_profile,
    get_user_by_public_id,
    is_onboarded,
    match_for_answers,
    pair_room_key,
    public_name,
    resolve_session,
    save_learner_profile,
    save_tutor_profile,
    set_display_name,
    set_user_role,
    style_from_teaching_answers,
)

from .auth import login_required

views = Blueprint("views", __name__)

_LEVELS = {o.value for o in GRADE_LEVELS}
_SUBJECTS = {o.value for o in SUBJECTS}
_LABELS = {o.value: o.label for o in GRADE_LEVELS + SUBJECTS}
MAX_RATE_SOL = 10
NAME_MIN, NAME_MAX = 2, 40


def _video_app_url():
    """Where the video app (live chat, calls, recordings) runs.

    The chat and call pages are served by this site but connect to the video
    app for everything live. A blank value falls back to the local default -
    copying .env.example leaves it set-but-empty, which must not break calls.
    """
    return (os.getenv("VIDEO_APP_URL") or "").strip().rstrip("/") or "http://localhost:8000"


# "/" is listed last so it registers first and becomes the canonical address -
# decorators apply bottom-up. Otherwise url_for("views.home") builds "/home".
@views.route("/home")
@views.route("/")
def home():
    # Public on purpose: logout returns here. Logged-in people who haven't set
    # up a profile yet are sent to onboarding, so nobody skips it.
    user = session.get("user")
    if user and not is_onboarded(user["sub"]):
        return redirect(url_for("views.onboarding"))
    return render_template("home.html")


# ---------------------------------------------------------------- onboarding


def _prefixed(form, prefix):
    """Form fields for one quiz. Both quizzes share question ids, so each is prefixed."""
    return {key[len(prefix):]: form.get(key) for key in form if key.startswith(prefix)}


def _unanswered(answers, questions):
    return [q.id for q in questions if not q.free_text and q.id not in answers]


def _values_from_profile(profile):
    """Pre-fill the form with what's already saved, so onboarding doubles as 'edit profile'."""
    if not profile:
        return {}
    user = profile["user"]
    name = public_name(user)
    values = {"display_name": "" if name == "TutorMatch member" else name}
    if user["role"] == "tutor" and profile["tutor"]:
        t = profile["tutor"]
        style = t["style_affinity"] or {}
        values.update({
            "role": "tutor",
            "subjects": t["subjects"] or [],
            "teaching_levels": t["teaching_levels"] or [],
            "bio": t["bio"] or "",
            "hourly_rate_sol": t["hourly_rate_sol"],
            # The picked answer is the one scored 1.0.
            **{f"teach_{axis}": max(scores, key=scores.get) for axis, scores in style.items() if scores},
        })
    elif profile["learner"]:
        l = profile["learner"]
        values.update({
            "role": "student",
            "subjects": l["subjects"] or [],
            "grade_level": l["grade_level"],
            **{f"learn_{k}": v for k, v in (l["raw_answers"] or {}).items()},
        })
    return values


@views.route("/onboarding", methods=["GET", "POST"])
@login_required
def onboarding():
    sub = session["user"]["sub"]
    errors = []

    if request.method == "POST":
        form = request.form
        role = form.get("role")
        subjects = [s for s in form.getlist("subjects") if s in _SUBJECTS]
        display_name = " ".join((form.get("display_name") or "").split())

        if not NAME_MIN <= len(display_name) <= NAME_MAX:
            errors.append(f"Choose a display name between {NAME_MIN} and {NAME_MAX} characters.")
        elif "@" in display_name:
            errors.append("Your display name is shown publicly, so it can't be an email address.")
        if role not in ("student", "tutor"):
            errors.append("Choose whether you're a student or a tutor.")
        if not subjects:
            errors.append("Pick at least one subject.")

        if role == "student":
            grade = form.get("grade_level")
            answers = clean_answers(_prefixed(form, "learn_"), QUIZ_QUESTIONS)
            if grade not in _LEVELS:
                errors.append("Choose your grade level.")
            if _unanswered(answers, QUIZ_QUESTIONS):
                errors.append("Answer every question in the learning style quiz.")
            if not errors:
                set_display_name(sub, display_name)
                set_user_role(sub, "student")
                save_learner_profile(
                    sub,
                    subjects=subjects,
                    pace=answers["pace"],
                    goals=answers.get("goal", ""),
                    raw_answers=answers,
                    profile_sentence=build_profile_sentence(answers, subjects),
                    grade_level=grade,
                )
                session["user"] = {**session["user"], "role": "student", "name": display_name}
                return redirect(url_for("views.matches"))

        elif role == "tutor":
            levels = [lvl for lvl in form.getlist("teaching_levels") if lvl in _LEVELS]
            answers = clean_answers(_prefixed(form, "teach_"), TEACHING_QUESTIONS)
            bio = (form.get("bio") or "").strip()[:1000]
            try:
                rate = round(float(form.get("hourly_rate_sol", "")), 4)
            except ValueError:
                rate = None
            if not levels:
                errors.append("Pick at least one level you teach.")
            if _unanswered(answers, TEACHING_QUESTIONS):
                errors.append("Answer every question in the teaching style quiz.")
            if len(bio) < 10:
                errors.append("Write a short bio (at least 10 characters).")
            if rate is None or not 0 < rate <= MAX_RATE_SOL:
                errors.append(f"Enter an hourly rate between 0 and {MAX_RATE_SOL} SOL.")
            if not errors:
                set_display_name(sub, display_name)
                set_user_role(sub, "tutor")
                save_tutor_profile(
                    sub,
                    bio=bio,
                    subjects=subjects,
                    teaching_levels=levels,
                    style_affinity=style_from_teaching_answers(answers),
                    hourly_rate_sol=rate,
                )
                session["user"] = {**session["user"], "role": "tutor", "name": display_name}
                return redirect(url_for("views.my_profile"))

        # Re-show the form with what they entered.
        values = {k: form.get(k) for k in form}
        values["subjects"] = form.getlist("subjects")
        values["teaching_levels"] = form.getlist("teaching_levels")
    else:
        values = _values_from_profile(get_profile(sub))

    return render_template(
        "onboarding.html",
        errors=errors,
        values=values,
        grade_levels=GRADE_LEVELS,
        subjects=SUBJECTS,
        quiz_questions=QUIZ_QUESTIONS,
        teaching_questions=TEACHING_QUESTIONS,
        max_rate=MAX_RATE_SOL,
        name_max=NAME_MAX,
    )


@views.route("/matches")
@login_required
def matches():
    profile = get_profile(session["user"]["sub"])
    if not profile or profile["user"]["role"] == "tutor":
        return redirect(url_for("views.home"))
    learner = profile["learner"]
    if not learner:
        return redirect(url_for("views.onboarding"))

    sentence, results = match_for_answers(
        learner["raw_answers"] or {},
        learner["subjects"] or [],
        grade_level=learner["grade_level"],
        limit=3,
    )
    return render_template(
        "matches.html",
        sentence=sentence,
        matches=results,
        grade_label=_LABELS.get(learner["grade_level"], ""),
        labels=_LABELS,
    )


# ------------------------------------------------------------------ profiles


def _onboarded_profile(user_row):
    """A user's full profile, or None if they haven't finished onboarding."""
    if not user_row:
        return None
    profile = get_profile(user_row["auth_sub"])
    key = "tutor" if profile["user"]["role"] == "tutor" else "learner"
    return profile if profile[key] else None


@views.route("/profile")
@login_required
def my_profile():
    profile = get_profile(session["user"]["sub"])
    if not _onboarded_profile(profile and profile["user"]):
        return redirect(url_for("views.onboarding"))
    return redirect(url_for("views.profile", public_id=profile["user"]["public_id"]))


@views.route("/profile/<public_id>")
@login_required
def profile(public_id):
    subject = _onboarded_profile(get_user_by_public_id(public_id))
    if subject is None:
        abort(404)
    user = subject["user"]
    me = session["user"]
    is_me = user["auth_sub"] == me["sub"]
    is_tutor = user["role"] == "tutor"
    details = subject["tutor"] if is_tutor else subject["learner"]

    if is_tutor:
        traits = describe_teaching_style(details["style_affinity"] or {})
        levels = details["teaching_levels"] or []
    else:
        traits = describe_learning_style(details["raw_answers"] or {})
        levels = [details["grade_level"]] if details["grade_level"] else []

    return render_template(
        "profile.html",
        person=user,
        name=public_name(user),
        is_me=is_me,
        is_tutor=is_tutor,
        details=details,
        traits=traits,
        subjects=[_LABELS.get(s, s) for s in details["subjects"] or []],
        levels=[_LABELS.get(l, l) for l in levels],
        # Chats pair a student with a tutor - the call needs exactly one of each.
        can_chat=not is_me and user["role"] != me.get("role"),
    )


# ------------------------------------------------------------- chat and call


def _pair_room(public_id):
    """Resolve the other person and the shared room, or abort if they can't be paired.

    Returns (their user row, room key, this user's role in the call).
    """
    me = get_profile(session["user"]["sub"])
    if not _onboarded_profile(me and me["user"]):
        abort(redirect(url_for("views.onboarding")))
    other = _onboarded_profile(get_user_by_public_id(public_id))
    if other is None:
        abort(404)
    me_user, other_user = me["user"], other["user"]
    if other_user["auth_sub"] == me_user["auth_sub"] or other_user["role"] == me_user["role"]:
        # The call's connection logic needs one tutor and one tutee.
        abort(400, "Chats connect a student with a tutor.")

    room = pair_room_key(me_user["auth_sub"], other_user["auth_sub"])
    # Record both people on the session now, so either can open its transcript.
    resolve_session(room)
    for person in (me_user, other_user):
        attach_participant(room, person["role"], person["auth_sub"])

    call_role = "tutor" if me_user["role"] == "tutor" else "tutee"
    return other_user, room, call_role


def _live_page(template, public_id):
    other, room, call_role = _pair_room(public_id)
    return render_template(
        template,
        other_name=public_name(other),
        other_public_id=other["public_id"],
        room=room,
        call_role=call_role,
        video_app_url=_video_app_url(),
    )


@views.route("/chat/<public_id>")
@login_required
def chat(public_id):
    return _live_page("chat.html", public_id)


@views.route("/call/<public_id>")
@login_required
def call(public_id):
    return _live_page("call.html", public_id)


@views.route("/ice-servers")
@login_required
def ice_servers():
    """STUN + TURN for the call page. Same Cloudflare credentials the video app uses."""
    return {"iceServers": get_ice_servers()}
